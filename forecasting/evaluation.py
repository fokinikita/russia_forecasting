"""Test metrics for backtest forecasts.

* summary        - RMSE / MAE / bias per specification, target and horizon,
                   relative to the naive forecast (last known y/y growth)
* ranking        - specifications ranked by RMSE relative to naive, averaged over horizons
* best           - best specification per target x horizon x availability
* errors         - every forecast with its fact and error (quality per test quarter)
* rolling        - RMSE over a moving window of test quarters (quality dynamics)
"""

from __future__ import annotations

import attrs
import polars as pl

from forecasting.dataset import QuarterlyDataset, shift_quarters, target_column
from forecasting.experiment import CONFIG_COLUMNS

SPEC_COLUMNS = [*CONFIG_COLUMNS, "target", "horizon"]


@attrs.define(slots=True)
class ForecastEvaluator:
    forecasts: pl.DataFrame
    dataset: QuarterlyDataset
    rolling_window: int = 4

    def _facts(self) -> pl.DataFrame:
        frame = self.dataset.frame
        return pl.concat(
            [
                frame.select(pl.col("date"), pl.lit(t).alias("target"), pl.col(target_column(t)).alias("value"))
                for t in self.dataset.targets
            ]
        ).drop_nulls("value")

    def errors(self) -> pl.DataFrame:
        facts = self._facts()
        return (
            self.forecasts.join(
                facts.rename({"date": "target_date", "value": "fact"}), on=["target", "target_date"], how="inner"
            )
            .with_columns(
                pl.col("origin")
                .map_elements(lambda d: shift_quarters(d, -1), return_dtype=pl.Date)
                .alias("_last_known")
            )
            .join(facts.rename({"date": "_last_known", "value": "naive"}), on=["target", "_last_known"], how="left")
            .drop("_last_known")
            .with_columns(
                (pl.col("pred") - pl.col("fact")).alias("error"),
                (pl.col("naive") - pl.col("fact")).alias("naive_error"),
            )
            .sort([*SPEC_COLUMNS, "target_date"])
        )

    @staticmethod
    def _aggregate(errors: pl.DataFrame, by: list[str]) -> pl.DataFrame:
        return (
            errors.group_by(by)
            .agg(
                pl.len().alias("n"),
                (pl.col("error") ** 2).mean().sqrt().alias("rmse"),
                pl.col("error").abs().mean().alias("mae"),
                pl.col("error").mean().alias("bias"),
                (pl.col("naive_error") ** 2).mean().sqrt().alias("rmse_naive"),
                pl.col("naive_error").abs().mean().alias("mae_naive"),
            )
            .with_columns(
                (pl.col("rmse") / pl.col("rmse_naive")).alias("rel_rmse_naive"),
                (pl.col("mae") / pl.col("mae_naive")).alias("rel_mae_naive"),
            )
        )

    def summary(self, errors: pl.DataFrame | None = None) -> pl.DataFrame:
        errors = self.errors() if errors is None else errors
        return self._aggregate(errors, SPEC_COLUMNS).sort(SPEC_COLUMNS)

    @staticmethod
    def ranking(summary: pl.DataFrame) -> pl.DataFrame:
        """Mean relative RMSE over horizons, per specification and target (lower is better)."""
        return (
            summary.group_by([*CONFIG_COLUMNS, "target"])
            .agg(
                pl.col("rel_rmse_naive").mean().alias("mean_rel_rmse"),
                pl.col("rmse").mean().alias("mean_rmse"),
                pl.len().alias("n_horizons"),
            )
            .with_columns(pl.col("mean_rel_rmse").rank("ordinal").over("target").alias("rank"))
            .sort(["target", "rank"])
        )

    @staticmethod
    def best(summary: pl.DataFrame) -> pl.DataFrame:
        """Best specification for every target, horizon and data availability."""
        return (
            summary.sort("rmse")
            .group_by(["target", "horizon", "availability"], maintain_order=True)
            .first()
            .sort(["target", "availability", "horizon"])
        )

    def rolling(self, errors: pl.DataFrame | None = None) -> pl.DataFrame:
        """RMSE of the last `rolling_window` test quarters, at every test quarter."""
        errors = self.errors() if errors is None else errors
        return (
            errors.sort("target_date")
            .with_columns(
                (pl.col("error") ** 2)
                .rolling_mean(self.rolling_window)
                .over(SPEC_COLUMNS)
                .sqrt()
                .alias("rolling_rmse"),
                (pl.col("naive_error") ** 2)
                .rolling_mean(self.rolling_window)
                .over(SPEC_COLUMNS)
                .sqrt()
                .alias("rolling_rmse_naive"),
            )
            .drop_nulls("rolling_rmse")
            .with_columns((pl.col("rolling_rmse") / pl.col("rolling_rmse_naive")).alias("rolling_rel_rmse_naive"))
            .select([*SPEC_COLUMNS, "target_date", "rolling_rmse", "rolling_rmse_naive", "rolling_rel_rmse_naive"])
            .sort([*SPEC_COLUMNS, "target_date"])
        )
