"""Direct multi-horizon design for the ML models.

For a forecast origin T (the current quarter, `availability` months of it known):

    X(T) = [monthly features of quarter T (month `availability`), target(T-1)]
    y(T) = target(T + h - 1)

h = 1 is the nowcast of quarter T, h = 8 is seven quarters after it.
This is the same information set as the original project (features lagged h-1,
target lagged h), but indexed by origin, so the same code produces backtest
and future forecasts.
"""

from __future__ import annotations

from datetime import date

import attrs
import numpy as np
import polars as pl

from forecasting.dataset import QuarterlyDataset, target_column

TARGET_LAG = "target_lag1"


@attrs.frozen
class DirectDesign:
    dataset: QuarterlyDataset
    target: str
    horizon: int
    feature_cols: tuple[str, ...]

    @property
    def x_columns(self) -> list[str]:
        return [*self.feature_cols, TARGET_LAG]

    def frame(self) -> pl.DataFrame:
        """origin | target_date | X columns | y   (one row per origin)."""
        y_col = target_column(self.target)
        frame = self.dataset.frame.sort("date")
        return frame.select(
            pl.col("date").alias("origin"),
            pl.col("date").dt.offset_by(f"{3 * (self.horizon - 1)}mo").alias("target_date"),
            # publication gaps: a missing value is replaced by the last known one (max 1 quarter)
            *[pl.col(c).forward_fill(limit=1) for c in self.feature_cols],
            pl.col(y_col).shift(1).alias(TARGET_LAG),
            pl.col(y_col).shift(-(self.horizon - 1)).alias("y"),
        )

    def training_rows(self, frame: pl.DataFrame, train_start_year: int, known_until: date) -> pl.DataFrame:
        """Rows usable for training when targets are known up to `known_until` (inclusive)."""
        return frame.filter(
            (pl.col("target_date").dt.year() >= train_start_year)
            & (pl.col("target_date") <= known_until)
            & pl.col("y").is_not_null()
            & pl.col(TARGET_LAG).is_not_null()
        )

    def to_numpy(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame.select(self.x_columns).to_numpy().astype(np.float64)
        y = frame["y"].to_numpy().astype(np.float64)
        return x, y
