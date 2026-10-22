"""Turn raw IEP series into model-ready "basic" series.

    * chain_mom  (% to previous period)     -> base index, 100 = period before first obs
    * chain_yoy  (% to same period last yr) -> index chained over 12 months, so that
                                               log(I_t) - log(I_{t-12}) == log(x_t / 100)
                                               (exactly what the d12 features use)
    * splice     -> one series out of several pieces: GDP components chained into
                    2021 prices via y/y growth of older base years, unemployment
    * daily      -> monthly average (miacr), like the old cbr_data
    * dates      -> period start (month / quarter start), like the old excel data
"""

from __future__ import annotations

import logging
from datetime import date

import attrs
import polars as pl

from data_iep.series_config import (
    DAILY_SERIES,
    MONTHLY_SERIES,
    QUARTERLY_SERIES,
    SeriesSpec,
)

logger = logging.getLogger(__name__)

_PERIODS_PER_YEAR = {"Q": 4, "M": 12}
_INTERVAL = {"Q": "1q", "M": "1mo"}


# --------------------------------------------------------------- primitives
def chain_mom_to_index(values: pl.Series) -> pl.Series:
    """% to previous period -> level index (100 = period before the first observation)."""
    if values.drop_nulls().is_empty():
        return values
    first, last = values.is_not_null().arg_true()[[0, -1]]
    if values[first : last + 1].null_count():
        logger.warning(f"{values.name}: gaps inside a chained series, index is broken after them")
    return (values / 100).cum_prod() * 100


def chain_yoy_to_index(values: pl.Series, periods_per_year: int) -> pl.Series:
    """% to same period of previous year -> index.

    The first year of observations is taken relative to 100, after that
    I_t = I_{t-p} * x_t / 100. Absolute level is arbitrary, yoy changes are exact.
    """
    raw = values.to_list()
    index: list[float | None] = [None] * len(raw)
    for t, x in enumerate(raw):
        if x is None:
            continue
        previous = index[t - periods_per_year] if t >= periods_per_year else None
        index[t] = (previous if previous is not None else 100.0) * x / 100
    return pl.Series(values.name, index, dtype=pl.Float64)


def splice_backwards(frame: pl.DataFrame, pieces: list[str], target: str, periods_per_year: int) -> pl.Series:
    """Chain pieces (newest first) into one series in the prices of the newest piece.

    Same method as the "Сцепка в пост. ценах" sheet: going back in time, every
    missing period is chained from the same period one year later using the
    y/y growth of the newest older piece that covers both periods:

        X[t] = X[t + p] * old[t] / old[t + p]      (p = 4 quarters / 12 months)

    So the result keeps the level of the newest base year (2021 prices) and the
    y/y growth rates of the older base years before it starts.
    """
    result = frame[pieces[0]].to_list()
    older = [frame[name].to_list() for name in pieces[1:]]

    for t in range(len(result) - periods_per_year - 1, -1, -1):
        if result[t] is not None or result[t + periods_per_year] is None:
            continue
        for piece in older:
            now, year_later = piece[t], piece[t + periods_per_year]
            if now is not None and year_later:
                result[t] = result[t + periods_per_year] * now / year_later
                break

    return pl.Series(target, result, dtype=pl.Float64)


def to_regular_grid(frame: pl.DataFrame, frequency: str) -> pl.DataFrame:
    """Make sure every period between min and max date is present (needed for shifts)."""
    grid = pl.DataFrame(
        {"date": pl.date_range(frame["date"].min(), frame["date"].max(), interval=_INTERVAL[frequency], eager=True)}
    )
    return grid.join(frame, on="date", how="left")


# ------------------------------------------------------------------ service
@attrs.define(slots=True)
class IepTransformService:
    monthly_start: date | None = date(1999, 1, 1)  # old datasets start here
    quarterly_start: date | None = None

    @staticmethod
    def _apply_specs(raw: pl.DataFrame, specs: list[SeriesSpec], frequency: str) -> pl.DataFrame:
        frame = to_regular_grid(raw, frequency)
        periods = _PERIODS_PER_YEAR[frequency]
        columns: list[pl.Series] = [frame["date"]]

        splice_targets: dict[str, list[str]] = {}
        for spec in specs:
            if spec.name not in frame.columns:
                logger.warning(f"{spec.name}: missing in raw data, skipped")
                continue
            if spec.kind == "splice":
                splice_targets.setdefault(spec.target, []).append(spec.name)
            elif spec.kind == "chain_mom":
                columns.append(chain_mom_to_index(frame[spec.name]))
            elif spec.kind == "chain_yoy":
                columns.append(chain_yoy_to_index(frame[spec.name], periods))
            else:
                columns.append(frame[spec.name])

        for target, pieces in splice_targets.items():
            columns.append(splice_backwards(frame, pieces, target, periods))

        return pl.DataFrame(columns)

    @staticmethod
    def _cut(frame: pl.DataFrame, start: date | None) -> pl.DataFrame:
        if start is not None:
            frame = frame.filter(pl.col("date") >= start)
        # drop leading/trailing periods where nothing is known
        has_data = pl.any_horizontal(pl.exclude("date").is_not_null())
        return frame.filter(has_data)

    def transform_daily(self, raw_daily: pl.DataFrame) -> pl.DataFrame:
        names = [s.name for s in DAILY_SERIES if s.name in raw_daily.columns]
        return raw_daily.select("date", *names).filter(pl.any_horizontal(pl.col(names).is_not_null())).sort("date")

    def daily_to_monthly(self, daily: pl.DataFrame) -> pl.DataFrame:
        return (
            daily.with_columns(pl.col("date").dt.truncate("1mo"))
            .group_by("date")
            .agg(pl.exclude("date").mean())
            .sort("date")
        )

    def transform_quarterly(self, raw_quarterly: pl.DataFrame) -> pl.DataFrame:
        frame = self._apply_specs(raw_quarterly, QUARTERLY_SERIES, "Q")
        return self._cut(frame, self.quarterly_start)

    def transform_monthly(self, raw_monthly: pl.DataFrame, raw_daily: pl.DataFrame | None = None) -> pl.DataFrame:
        frame = self._apply_specs(raw_monthly, MONTHLY_SERIES, "M")
        if raw_daily is not None and raw_daily.height:
            monthly_from_daily = self.daily_to_monthly(self.transform_daily(raw_daily))
            frame = frame.join(monthly_from_daily, on="date", how="left")
        return self._cut(frame, self.monthly_start)

    def run(
        self, raw_quarterly: pl.DataFrame, raw_monthly: pl.DataFrame, raw_daily: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """-> quarterly, monthly (incl. monthly means of daily series), daily."""
        return (
            self.transform_quarterly(raw_quarterly),
            self.transform_monthly(raw_monthly, raw_daily),
            self.transform_daily(raw_daily),
        )
