"""Static HTML dashboard with the results of a real-time run.

The page (dashboard/template.html, same design as the business mock-up) gets all numbers embedded
as JSON, so the produced file opens in any browser without a server:

    python -m pipelines.build_dashboard                 # latest finished run -> dashboard/macro_dashboard.html
    python -m pipelines.build_dashboard --run-id 2026-10-22 --as-of 2026-11-15

Data availability shown by default follows the calendar: in the 1st month of a quarter only one month
of the origin quarter is assumed to be published (availability 1), in the 2nd month two, in the 3rd three.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import attrs
import numpy as np
import polars as pl

import config
from forecasting.experiment import CONFIG_COLUMNS
from forecasting.report import RunResults

logger = logging.getLogger(__name__)

SPEC_KEY_COLUMNS = ["model", "train_start", "var_set", "features_type", "availability"]
TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "template.html"
DATA_PLACEHOLDER = "/*__DASHBOARD_DATA__*/null"
# JS libraries are inlined from local copies, so the page works offline as a single file
VENDOR_DIR = TEMPLATE_PATH.parent / "vendor"
VENDOR_SCRIPTS = {
    "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js": "chart.umd.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js": "xlsx.full.min.js",
}


def _inline_scripts(html: str) -> str:
    for url, name in VENDOR_SCRIPTS.items():
        tag = f'<script src="{url}"></script>'
        path = VENDOR_DIR / name
        if tag not in html:
            continue
        if path.exists():
            code = path.read_text(encoding="utf-8").replace("</script", "<\\/script")
            html = html.replace(tag, f"<script>/* {name} */\n{code}\n</script>")
        else:
            logger.warning(f"{path} not found, the page will load {url} from the internet")
    return html

TARGET_NAMES = {
    "gdp": "ВВП в постоянных ценах",
    "cons": "Потребление домашних хозяйств в постоянных ценах",
    "inv": "Валовое накопление в постоянных ценах",
    "inv_cap": "Валовое накопление основного капитала в постоянных ценах",
}


def availability_for(as_of: date) -> int:
    """Months of the current quarter published by `as_of`: 1st month of a quarter -> 1, ..., 3rd -> 3."""
    return (as_of.month - 1) % 3 + 1


def _spec_key() -> pl.Expr:
    return pl.concat_str([pl.col(c).cast(pl.Utf8) for c in SPEC_KEY_COLUMNS], separator="/").alias("key")


def _round(values, digits: int = 5) -> list:
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), digits) for v in values]


@attrs.define
class DashboardBuilder:
    """Collects history, test metrics and future forecasts of one run into the dashboard payload."""

    run: RunResults
    as_of: date = attrs.field(factory=date.today)
    top_k: int = config.RT_CONSENSUS_TOP_K
    history_start: int = 2000

    # ------------------------------------------------------------ inputs
    def _levels(self) -> pl.DataFrame:
        return pl.read_parquet(Path(config.IEP_DATA_DIR) / "quarterly.parquet").sort("date")

    @staticmethod
    def _n_all_variables() -> int:
        """Number of monthly / daily IEP indicators that form the "all" variable set."""
        names = set()
        for name in ("monthly.parquet", "daily.parquet"):
            path = Path(config.IEP_DATA_DIR) / name
            if path.exists():
                names |= set(pl.read_parquet_schema(path)) - {"date"}
        return len(names)

    def _errors(self) -> pl.DataFrame:
        return self.run.errors.with_columns(_spec_key())

    def _future(self) -> pl.DataFrame:
        return self.run.future.with_columns(_spec_key())

    # ----------------------------------------------------------- metrics
    @staticmethod
    def _metrics_by_horizon(errors: pl.DataFrame, by: list[str]) -> pl.DataFrame:
        """MAE / RMSE in p.p. of y/y growth and MAPE of the implied level (level error = exp(log error) - 1)."""
        return errors.group_by([*by, "horizon"]).agg(
            (pl.col("error").abs().mean() * 100).alias("mae"),
            ((pl.col("error") ** 2).mean().sqrt() * 100).alias("rmse"),
            ((pl.col("error").exp() - 1).abs().mean() * 100).alias("mape"),
            pl.len().alias("n"),
        )

    def _consensus_errors(self, errors: pl.DataFrame) -> pl.DataFrame:
        """Backtest of the consensus: median of the top-K specs (by RMSE) of every target x availability x horizon."""
        metrics = self.run.metrics.select(*CONFIG_COLUMNS, "target", "horizon", "rmse")
        top = metrics.with_columns(
            pl.col("rmse").rank("ordinal").over(["target", "availability", "horizon"]).alias("_rank")
        ).filter(pl.col("_rank") <= self.top_k)
        return (
            errors.join(top.select(*CONFIG_COLUMNS, "target", "horizon"), on=[*CONFIG_COLUMNS, "target", "horizon"])
            .group_by(["target", "availability", "horizon", "target_date"])
            .agg(pl.col("pred").median(), pl.col("fact").first())
            .with_columns(
                (pl.col("pred") - pl.col("fact")).alias("error"),
                pl.concat_str([pl.lit("consensus"), pl.col("availability").cast(pl.Utf8)], separator="/").alias(
                    "key"
                ),
            )
        )

    def _best_errors(self, errors: pl.DataFrame) -> pl.DataFrame:
        """Backtest of the best specification: the top-ranked model with its auto-selected train start,
        variable set and features type (one spec for all horizons, backtest/selected_specs.csv).
        Runs made before the automatic selection existed fall back to the best spec per horizon."""
        selected = self.run.selected_specs
        if selected is None:
            best = self.run.best.select(*CONFIG_COLUMNS, "target", "horizon")
            joined = errors.join(best, on=[*CONFIG_COLUMNS, "target", "horizon"])
        else:
            chosen = selected.filter(pl.col("model_rank") == 1).select(*CONFIG_COLUMNS, "target")
            joined = errors.join(chosen, on=[*CONFIG_COLUMNS, "target"])
        return joined.with_columns(
            pl.concat_str([pl.lit("best"), pl.col("availability").cast(pl.Utf8)], separator="/").alias("key")
        )

    # ------------------------------------------------------------ payload
    def build(self) -> dict:
        info = self.run.info
        horizon = int(info["settings"]["horizon"])
        errors = self._errors()
        future = self._future()
        test_dates = sorted(errors["target_date"].unique().to_list())
        date_index = {d: i for i, d in enumerate(test_dates)}

        consensus_errors = self._consensus_errors(errors)
        best_errors = self._best_errors(errors)
        all_errors = pl.concat(
            [
                errors.select("key", "target", "horizon", "target_date", "pred", "error"),
                consensus_errors.select("key", "target", "horizon", "target_date", "pred", "error"),
                best_errors.select("key", "target", "horizon", "target_date", "pred", "error"),
            ]
        )

        targets = list(info["settings"]["targets"])
        payload_targets = {}
        levels = self._levels()
        for target in targets:
            history = levels.select("date", target).drop_nulls().filter(pl.col("date").dt.year() >= self.history_start)
            payload_targets[target] = {
                "name": TARGET_NAMES.get(target, target),
                "history": [[d.isoformat(), round(v, 1)] for d, v in history.iter_rows()],
                "test_fact": [None] * len(test_dates),
                "naive": {},
                "specs": {},
            }

        # test facts and naive forecasts (identical for every spec)
        naive = errors.select("target", "horizon", "target_date", "fact", "naive", "naive_error").unique(
            ["target", "horizon", "target_date"]
        )
        for row in naive.iter_rows(named=True):
            t = payload_targets[row["target"]]
            t["test_fact"][date_index[row["target_date"]]] = round(row["fact"], 5)
            t["naive"].setdefault(str(row["horizon"]), {"pred": [None] * len(test_dates)})
            t["naive"][str(row["horizon"])]["pred"][date_index[row["target_date"]]] = round(row["naive"], 5)
        naive_metrics = self._metrics_by_horizon(naive.rename({"naive_error": "error"}), ["target"])
        for row in naive_metrics.iter_rows(named=True):
            payload_targets[row["target"]]["naive"][str(row["horizon"])].update(
                {k: round(row[k], 4) for k in ("mae", "rmse", "mape")}
            )

        def spec_entry(target: str, key: str) -> dict:
            specs = payload_targets[target]["specs"]
            if key not in specs:
                specs[key] = {
                    "future": [None] * horizon,
                    "metrics": [None] * horizon,
                    "test": [[None] * len(test_dates) for _ in range(horizon)],
                }
            return specs[key]

        # backtest predictions
        for key, target, h, target_date, pred, _ in all_errors.iter_rows():
            spec_entry(target, key)["test"][h - 1][date_index[target_date]] = round(pred, 5)
        for row in self._metrics_by_horizon(all_errors, ["key", "target"]).iter_rows(named=True):
            spec_entry(row["target"], row["key"])["metrics"][row["horizon"] - 1] = {
                k: round(row[k], 4) for k in ("mae", "rmse", "mape")
            } | {"n": row["n"]}

        # future forecasts
        future_pred = {}
        for key, target, h, pred in future.select("key", "target", "horizon", "pred").iter_rows():
            spec_entry(target, key)["future"][h - 1] = round(pred, 5)
            future_pred[(target, key, h)] = pred
        for row in self.run.future_consensus.iter_rows(named=True):
            entry = spec_entry(row["target"], f"consensus/{row['availability']}")
            entry["future"][row["horizon"] - 1] = round(row["pred"], 5)
            # mean of the top-K member forecasts (the consensus itself is their median)
            member_preds = [
                future_pred[(row["target"], spec, row["horizon"])]
                for spec in row["specs"].split(";")
                if (row["target"], spec, row["horizon"]) in future_pred
            ]
            entry.setdefault("mean", [None] * horizon)[row["horizon"] - 1] = (
                round(float(np.mean(member_preds)), 5) if member_preds else None
            )
            # every member forecast: the page draws the 25-75% range of them (min-max is too wide)
            entry.setdefault("member_preds", [None] * horizon)[row["horizon"] - 1] = sorted(
                round(float(v), 5) for v in member_preds
            )
            entry.setdefault("band", [None] * horizon)[row["horizon"] - 1] = [
                round(row["min_yoy_pct"], 3),
                round(row["max_yoy_pct"], 3),
            ]
            entry.setdefault("members", [None] * horizon)[row["horizon"] - 1] = row["specs"]
        future_selected = self.run.future_selected
        best_future = (
            self.run.future_best if future_selected is None else future_selected.filter(pl.col("model_rank") == 1)
        )
        for row in best_future.iter_rows(named=True):
            entry = spec_entry(row["target"], f"best/{row['availability']}")
            entry["future"][row["horizon"] - 1] = round(row["pred"], 5)
            spec = "/".join(str(row[c]) for c in CONFIG_COLUMNS)
            entry.setdefault("members", [None] * horizon)[row["horizon"] - 1] = spec
        selected = self.run.selected_specs
        if selected is not None:
            # the chosen spec is shown in the sample / variable set / features controls
            for row in selected.filter(pl.col("model_rank") == 1).iter_rows(named=True):
                entry = spec_entry(row["target"], f"best/{row['availability']}")
                entry["chosen"] = "/".join(str(row[c]) for c in CONFIG_COLUMNS)

        origin = future["origin"].min()
        return {
            "meta": {
                "run_id": info["run_id"],
                "created": info["created"],
                "last_known_quarter": info["last_known_quarter"],
                "origin": str(origin),
                "as_of": self.as_of.isoformat(),
                "actual_availability": availability_for(self.as_of),
                "horizon": horizon,
                "top_k": self.top_k,
                "variable_sets": info["settings"].get("variable_sets", {}),
                "n_all_variables": self._n_all_variables(),
                "built": datetime.now().isoformat(timespec="minutes"),
            },
            "test_dates": [d.isoformat() for d in test_dates],
            "targets": payload_targets,
        }

    def write(self, out_path: Path | str = Path("dashboard") / "macro_dashboard.html") -> Path:
        out_path = Path(out_path)
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        if DATA_PLACEHOLDER not in template:
            raise ValueError(f"{TEMPLATE_PATH} has no data placeholder {DATA_PLACEHOLDER!r}")
        payload = json.dumps(self.build(), ensure_ascii=False, separators=(",", ":"), default=str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html = _inline_scripts(template).replace(DATA_PLACEHOLDER, payload)
        out_path.write_text(html, encoding="utf-8")
        logger.info(f"dashboard saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path


def build_dashboard(
    run_id: Optional[str] = None,
    as_of: Optional[date] = None,
    out_path: Path | str = Path("dashboard") / "macro_dashboard.html",
    results_dir: str = config.RT_RESULTS_DIR,
) -> Path:
    run = RunResults(Path(results_dir) / "runs" / run_id) if run_id else RunResults.latest(results_dir)
    return DashboardBuilder(run, as_of=as_of or date.today()).write(out_path)
