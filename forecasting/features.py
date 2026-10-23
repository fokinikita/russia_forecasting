"""Monthly feature transformation (the same as the original project used).

* positive series   -> log difference to the same month of the previous year: `<var>_log_d12`
* other series      -> plain difference to the same month of the previous year: `<var>_d12`
* rolling features  -> rolling means of the above over `windows` months: `<...>_d12_roll_mean_<w>`
"""

from __future__ import annotations

import attrs
import polars as pl

import config


@attrs.define(slots=True)
class MonthlyFeatures:
    windows: tuple[int, ...] = tuple(config.ROLLING_WINDOWS_MONTH)
    date_column: str = "date"
    columns_d12: list[str] = attrs.field(factory=list)
    columns_rolling: list[str] = attrs.field(factory=list)

    def transform(self, monthly: pl.DataFrame) -> pl.DataFrame:
        """date | <var>... -> date | d12 features | rolling features (column lists kept in the object)."""
        monthly = monthly.sort(self.date_column)
        variables = [c for c in monthly.columns if c != self.date_column]
        positive = [c for c in variables if monthly.filter(pl.col(c) <= 0).height == 0]
        other = [c for c in variables if c not in positive]

        d12 = [(pl.col(c).log() - pl.col(c).log().shift(12)).alias(f"{c}_log_d12") for c in positive]
        d12 += [(pl.col(c) - pl.col(c).shift(12)).alias(f"{c}_d12") for c in other]
        self.columns_d12 = [f"{c}_log_d12" for c in positive] + [f"{c}_d12" for c in other]

        rolling = [
            pl.col(c).rolling_mean(window).alias(f"{c}_roll_mean_{window}")
            for window in self.windows
            for c in self.columns_d12
        ]
        self.columns_rolling = [f"{c}_roll_mean_{w}" for w in self.windows for c in self.columns_d12]

        return monthly.with_columns(d12).with_columns(rolling).select(
            self.date_column, *self.columns_d12, *self.columns_rolling
        )
