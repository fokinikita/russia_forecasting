from datetime import date

import numpy as np
import pytest

from forecasting.readme_charts import END_MARK, START_MARK, ReadmeCharts


def _spec(future, mae, members=None):
    test = [[0.01, 0.02] for _ in future]
    return {
        "future": future,
        "metrics": [{"mae": mae, "rmse": mae, "mape": mae, "n": 2} for _ in future],
        "test": test,
        **({"member_preds": members} if members else {}),
    }


def _payload() -> dict:
    quarters = [date(2024 + m // 12, m % 12 + 1, 1) for m in range(0, 24, 3)]  # 2024Q1 .. 2025Q4
    history = [[d.isoformat(), 100.0 + 10 * i] for i, d in enumerate(quarters)]
    growth = [np.log(1.02), np.log(1.03)]
    return {
        "meta": {"actual_availability": 1, "origin": "2026-01-01", "created": "2026-10-23T19:14", "top_k": 10},
        "test_dates": ["2025-07-01", "2025-10-01"],
        "targets": {
            "gdp": {
                "name": "ВВП",
                "history": history,
                "test_fact": [0.01, 0.02],
                "naive": {"1": {"mae": 1.0}, "2": {"mae": 1.0}},
                "specs": {
                    "consensus/1": _spec(growth, 0.5, members=[[0.0, 0.01, 0.03], None]),
                    "gb/2000/short/classic/1": _spec(growth, 0.9),
                    "gb/2010/all/rolling/1": _spec(growth, 0.7),
                    "gb/2010/all/rolling/2": _spec(growth, 0.1),  # other availability: ignored
                },
            }
        },
    }


def test_forecast_values():
    payload = _payload()
    fc = ReadmeCharts(payload, n_fact=2).forecast(payload["targets"]["gdp"], "consensus")
    assert [d for d, _ in fc.fact_yoy] == [date(2025, 7, 1), date(2025, 10, 1)]
    assert fc.fact_yoy[-1][1] == pytest.approx((170 / 130 - 1) * 100)
    (d1, g1, lo1, hi1, lvl1), (_, g2, lo2, hi2, _) = fc.future
    assert d1 == date(2026, 1, 1)
    assert (g1, g2) == pytest.approx((2.0, 3.0))
    assert lvl1 == pytest.approx(140 * 1.02)  # level a year ago x growth
    assert lo1 == pytest.approx((np.exp(0.005) - 1) * 100) and hi1 == pytest.approx((np.exp(0.02) - 1) * 100)
    assert lo2 == hi2 == pytest.approx(3.0)  # no members: range collapses to the forecast
    assert "26Q1*" in fc.yoy_chart() and "xychart-beta" in fc.level_chart() and "h = 1" in fc.backtest_chart()


def test_single_model_takes_spec_with_lowest_mae():
    payload = _payload()
    assert ReadmeCharts(payload).spec_key(payload["targets"]["gdp"], "gb") == "gb/2010/all/rolling/1"
    assert ReadmeCharts(payload).spec_key(payload["targets"]["gdp"], "dfm") is None


def test_write_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"head\n{START_MARK}\nold\n{END_MARK}\ntail\n", encoding="utf-8")
    ReadmeCharts(_payload()).write_readme(readme)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and text.count("```mermaid") == 6  # consensus + CatBoost, 3 tabs each
    assert text.startswith("head\n") and text.endswith("tail\n")
