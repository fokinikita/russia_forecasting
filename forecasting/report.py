"""Read the results of a real-time run (see forecasting/pipeline.py for the layout)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import attrs
import polars as pl

import config
from forecasting.experiment import CONFIG_COLUMNS

DESIGN_FACTORS = ("model", "train_start", "var_set", "features_type", "availability")


@attrs.define(slots=True)
class RunResults:
    run_dir: Path = attrs.field(converter=Path)

    @classmethod
    def latest(cls, results_dir: str = config.RT_RESULTS_DIR) -> "RunResults":
        runs = sorted(
            p
            for p in (Path(results_dir) / "runs").iterdir()
            if (p / "backtest" / "metrics.csv").exists() and (p / "future" / "forecasts.csv").exists()
        )
        if not runs:
            raise FileNotFoundError(f"no finished runs in {results_dir}/runs")
        return cls(runs[-1])

    # ------------------------------------------------------------ loading
    def _csv(self, *parts: str) -> pl.DataFrame:
        return pl.read_csv(self.run_dir.joinpath(*parts), try_parse_dates=True)

    @property
    def info(self) -> dict:
        return json.loads((self.run_dir / "run_info.json").read_text(encoding="utf-8"))

    @property
    def metrics(self) -> pl.DataFrame:
        return self._csv("backtest", "metrics.csv")

    @property
    def ranking(self) -> pl.DataFrame:
        return self._csv("backtest", "ranking.csv")

    @property
    def best(self) -> pl.DataFrame:
        return self._csv("backtest", "best.csv")

    @property
    def selected_specs(self) -> Optional[pl.DataFrame]:
        path = self.run_dir / "backtest" / "selected_specs.csv"
        return pl.read_csv(path) if path.exists() else None

    @property
    def future_selected(self) -> Optional[pl.DataFrame]:
        path = self.run_dir / "future" / "selected.csv"
        return pl.read_csv(path, try_parse_dates=True) if path.exists() else None

    @property
    def rolling(self) -> pl.DataFrame:
        return self._csv("backtest", "rolling.csv")

    @property
    def errors(self) -> pl.DataFrame:
        return pl.read_parquet(self.run_dir / "backtest" / "errors.parquet")

    @property
    def future(self) -> pl.DataFrame:
        return self._csv("future", "forecasts.csv")

    @property
    def future_best(self) -> pl.DataFrame:
        return self._csv("future", "best.csv")

    @property
    def future_consensus(self) -> pl.DataFrame:
        return self._csv("future", "consensus.csv")

    @property
    def failed_specs(self) -> Optional[pl.DataFrame]:
        path = self.run_dir / "backtest" / "failed_specs.csv"
        return pl.read_csv(path) if path.exists() else None

    # ----------------------------------------------------------- analysis
    def best_by_model(self) -> pl.DataFrame:
        """Lowest RMSE per model, target, horizon (any other design choice)."""
        return (
            self.metrics.sort("rmse")
            .group_by(["model", "target", "horizon"], maintain_order=True)
            .first()
            .sort(["target", "model", "horizon"])
        )

    def factor_effect(self, factor: str) -> pl.DataFrame:
        """Median relative RMSE for each level of one design factor, per model and target."""
        if factor not in DESIGN_FACTORS:
            raise ValueError(f"factor must be one of {DESIGN_FACTORS}")
        by = ["model", "target", factor] if factor != "model" else ["model", "target"]
        return self.metrics.group_by(by).agg(pl.col("rel_rmse_naive").median().alias("median_rel_rmse")).sort(by)

    def share_beating_naive(self) -> pl.DataFrame:
        return (
            self.metrics.group_by(["model", "target"])
            .agg((pl.col("rel_rmse_naive") < 1).mean().alias("share_specs_better_than_naive"))
            .sort(["target", "model"])
        )

    def top_specs(self, target: str, n: int = 10) -> pl.DataFrame:
        return (
            self.ranking.filter(pl.col("target") == target)
            .head(n)
            .select("rank", *CONFIG_COLUMNS, "mean_rel_rmse", "mean_rmse")
        )
