"""MF-BVAR wrapper without R: data layout, fallback to previous origins, spec selection.

    python -m pytest tests -q
"""

import sys
from datetime import date

import attrs
import polars as pl
import pytest

from forecasting.dataset import QuarterlyDataset, shift_quarters
from forecasting.evaluation import ForecastEvaluator
from forecasting.mfbvar import MfbvarError, MfbvarForecaster, MfbvarInputs, MfbvarSettings

HORIZON = 4


def make_dataset(n_quarters: int = 40) -> QuarterlyDataset:
    dates = [shift_quarters(date(2010, 1, 1), k) for k in range(n_quarters)]
    frame = pl.DataFrame(
        {
            "date": dates,
            "gdp_log_d4": [0.01 * k for k in range(n_quarters)],
            **{f"x_log_d12_m{m}": [100 * k + m for k in range(n_quarters)] for m in (1, 2, 3)},
            # starts 3 years later than x -> dropped for train_start 2010 (max_late_start_years = 2)
            **{f"late_log_d12_m{m}": [None] * 12 + [1.0] * (n_quarters - 12) for m in (1, 2, 3)},
        },
        schema_overrides={f"late_log_d12_m{m}": pl.Float64 for m in (1, 2, 3)},
    )
    return QuarterlyDataset(frame=frame, targets=("gdp",), variables=("x", "late"))


@attrs.define(slots=True)
class FakeForecaster(MfbvarForecaster):
    """Pretends to be R: origins in `failing` raise, others forecast `pred = origin index`."""

    failing: frozenset = frozenset()
    calls: list = attrs.field(factory=list)

    def _run_batch(self, origins):
        self.calls.append(list(origins))
        n_quarters = self.horizon + self.settings.max_fallback
        for origin in origins:
            if origin in self.failing:
                self.failures[origin] = "inv_sympd(): matrix is singular"
                continue
            self.estimates[origin] = pl.DataFrame(
                {
                    "origin": [origin] * n_quarters,
                    "target": ["gdp_log_d4"] * n_quarters,
                    "target_date": [shift_quarters(origin, k) for k in range(n_quarters)],
                    "pred": [float(origin.year * 10 + origin.month)] * n_quarters,
                }
            )


def forecaster(failing=()) -> FakeForecaster:
    settings = MfbvarSettings(rscript=sys.executable, max_fallback=2)
    return FakeForecaster(
        make_dataset(), ["x"], "classic", 2010, availability=2, horizon=HORIZON, settings=settings, failing=frozenset(failing)
    )


def test_monthly_layout_and_late_series_dropped():
    inputs = MfbvarInputs(make_dataset(), ["x", "late"], "classic", 2010)
    monthly, quarterly, start = inputs.build()
    assert inputs.dropped == ["late_log_d12"]
    assert monthly.columns == ["date", "x_log_d12"]
    assert start == date(2010, 1, 1)
    # month m of quarter k -> value 100 k + m
    assert monthly.filter(pl.col("date") == date(2010, 5, 1))["x_log_d12"].item() == 102
    assert quarterly.columns == ["date", "gdp_log_d4"]


def test_successful_origins_are_used_directly():
    f = forecaster()
    origins = [date(2015, 1, 1), date(2015, 4, 1)]
    out = f.forecast(origins)
    assert out.height == len(origins) * HORIZON
    assert (out["refit_date"] == out["origin"]).all()
    assert sorted(out["horizon"].unique().to_list()) == list(range(1, HORIZON + 1))
    assert out["target"].unique().to_list() == ["gdp"]


def test_failed_origin_takes_previous_forecast():
    failed = date(2015, 4, 1)
    f = forecaster(failing=[failed])
    out = f.forecast([date(2015, 1, 1), failed, date(2015, 7, 1)])
    rows = out.filter(pl.col("origin") == failed).sort("horizon")
    assert rows.height == HORIZON
    assert (rows["refit_date"] == date(2015, 1, 1)).all()
    assert rows["target_date"].to_list() == [shift_quarters(failed, h) for h in range(HORIZON)]
    assert (rows["pred"] == 2015 * 10 + 1).all()


def test_fallback_estimates_missing_previous_origin_lazily():
    failed = date(2016, 1, 1)
    f = forecaster(failing=[failed])
    out = f.forecast([failed])  # the previous origin was not requested -> estimated on demand
    assert f.calls == [[failed], [date(2015, 10, 1)]]
    assert (out["refit_date"] == date(2015, 10, 1)).all() and out.height == HORIZON


def test_point_skipped_when_no_fallback_left():
    f = forecaster(failing=[date(2016, 1, 1), date(2015, 10, 1), date(2015, 7, 1)])  # max_fallback = 2
    out = f.forecast([date(2016, 1, 1), date(2016, 4, 1)])
    assert out["origin"].unique().to_list() == [date(2016, 4, 1)]


def test_all_failed_raises():
    f = forecaster(failing=[shift_quarters(date(2016, 1, 1), -k) for k in range(5)])
    with pytest.raises(MfbvarError):
        f.forecast([date(2016, 1, 1)])


def test_selection_per_model_target_availability():
    rows = []
    for model, start, var_set, offset in [
        ("gb", 2000, "short", 0.3),
        ("gb", 2010, "all", 0.1),  # best gb spec
        ("mfbvar", 2010, "short", 0.2),
        ("mfbvar", 2000, "all", 0.05),  # better on average but misses horizon 2 -> not chosen
    ]:
        for h in (1, 2):
            if model == "mfbvar" and var_set == "all" and h == 2:
                continue
            rows.append(
                dict(model=model, train_start=start, var_set=var_set, features_type="classic", availability=1,
                     target="gdp", horizon=h, n=12, rmse=offset, mae=offset, rel_rmse_naive=offset + 0.01 * h)
            )  # fmt: skip
    selected = ForecastEvaluator.selected(pl.DataFrame(rows))
    chosen = {r["model"]: (r["train_start"], r["var_set"], r["model_rank"]) for r in selected.iter_rows(named=True)}
    assert chosen == {"gb": (2010, "all", 1), "mfbvar": (2010, "short", 2)}
