"""Quarterly modelling table built from the IEP data.

One row per quarter (quarter start date):

    date | gdp_log_d4 ... | <var>_log_d12_m1 | <var>_log_d12_m2 | ... | <var>_log_d12_roll_mean_3_m1 | ...

* targets  - log y/y growth of the quarterly series (`<target>_log_d4`)
* features - monthly variables transformed exactly like FeaturesService does
  (log-d12 or d12, plus rolling means of them), split by month of the quarter.
  `_m{a}` = value in month `a` of the quarter, so "availability a" = the first
  `a` months of the quarter are published.

Rows go up to the last quarter that has any monthly data, so the table already
contains the "future" quarters needed for real-time forecasts.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional

import attrs
import polars as pl

import config
from preprocess_data.prepare_data import FeaturesService

logger = logging.getLogger(__name__)

FEATURE_TYPES = ("classic", "rolling")
_FEATURE_RE = re.compile(r"^(?P<var>.+?)(?:_log)?_d12(?P<roll>_roll_mean_\d+)?_m(?P<month>[123])$")


def target_column(target: str) -> str:
    return f"{target}_log_d4"


def parse_feature(column: str) -> Optional[tuple[str, str, int]]:
    """'brent_log_d12_roll_mean_3_m2' -> ('brent', 'rolling', 2); None for non-features."""
    match = _FEATURE_RE.match(column)
    if match is None:
        return None
    features_type = "rolling" if match["roll"] else "classic"
    return match["var"], features_type, int(match["month"])


def shift_quarters(day: date, quarters: int) -> date:
    month_index = day.year * 12 + (day.month - 1) + 3 * quarters
    return date(month_index // 12, month_index % 12 + 1, 1)


@attrs.frozen
class QuarterlyDataset:
    frame: pl.DataFrame  # date + targets + features on a regular quarterly grid
    targets: tuple[str, ...]  # short names: gdp, cons, ...
    variables: tuple[str, ...]  # monthly variables available in the features

    # ----------------------------------------------------------- columns
    @property
    def target_columns(self) -> list[str]:
        return [target_column(t) for t in self.targets]

    def feature_columns(
        self, features_type: str, availability: int, variables: Optional[list[str]] = None
    ) -> list[str]:
        """Feature columns of one type, for month `availability` of the quarter, for the given variables."""
        if features_type not in FEATURE_TYPES:
            raise ValueError(f"features_type must be one of {FEATURE_TYPES}, got {features_type!r}")
        wanted = set(variables) if variables is not None else None
        columns = []
        for column in self.frame.columns:
            parsed = parse_feature(column)
            if parsed is None:
                continue
            var, ftype, month = parsed
            if ftype == features_type and month == availability and (wanted is None or var in wanted):
                columns.append(column)
        return columns

    def resolve_variable_set(self, name: str, variables: Optional[list[str]]) -> list[str]:
        """Variables of a set that really exist in the data (missing ones are logged, not fatal)."""
        if variables is None:
            return list(self.variables)
        missing = [v for v in variables if v not in self.variables]
        if missing:
            logger.warning(f"variable set {name!r}: no data for {missing}, they are skipped")
        present = [v for v in variables if v in self.variables]
        if not present:
            raise ValueError(f"variable set {name!r} has no variables with data")
        return present

    # ------------------------------------------------------------- dates
    @property
    def last_target_date(self) -> date:
        """Last quarter where all targets are known."""
        known = self.frame.filter(pl.all_horizontal(pl.col(self.target_columns).is_not_null()))
        return known["date"].max()

    def test_dates(self, test_len: int) -> list[date]:
        last = self.last_target_date
        return [shift_quarters(last, -k) for k in range(test_len - 1, -1, -1)]


@attrs.define(slots=True)
class DatasetBuilder:
    """IEP monthly + quarterly frames -> QuarterlyDataset."""

    targets: tuple[str, ...] = tuple(config.RT_TARGETS)

    @staticmethod
    def _monthly_features(monthly: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
        service = FeaturesService()
        transformed = service._get_features_monthly(monthly.rename({"date": "datem"}).sort("datem"))
        columns = service.columns_d12 + service.columns_rolling
        return transformed.select(["datem", *columns]).rename({"datem": "date"}), columns

    @staticmethod
    def _split_by_month_of_quarter(features: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        with_quarter = features.with_columns(
            ((pl.col("date").dt.month() - 1) % 3 + 1).alias("_month"),
            pl.col("date").dt.truncate("1q").alias("_quarter"),
        )
        result = None
        for month in (1, 2, 3):
            part = with_quarter.filter(pl.col("_month") == month).select(
                pl.col("_quarter").alias("date"),
                *[pl.col(c).alias(f"{c}_m{month}") for c in columns],
            )
            result = part if result is None else result.join(part, on="date", how="full", coalesce=True)
        return result

    def _targets(self, quarterly: pl.DataFrame) -> pl.DataFrame:
        frame = quarterly.select(["date", *self.targets]).sort("date")
        return frame.select(
            "date",
            *[(pl.col(t).log() - pl.col(t).log().shift(4)).alias(target_column(t)) for t in self.targets],
        )

    def build(self, monthly: pl.DataFrame, quarterly: pl.DataFrame) -> QuarterlyDataset:
        monthly_features, columns = self._monthly_features(monthly)
        quarterly_features = self._split_by_month_of_quarter(monthly_features, columns)
        targets = self._targets(quarterly)

        joined = targets.join(quarterly_features, on="date", how="full", coalesce=True).sort("date")
        grid = pl.DataFrame({"date": pl.date_range(joined["date"].min(), joined["date"].max(), "1q", eager=True)})
        frame = grid.join(joined, on="date", how="left")

        # drop quarters without any information at all
        frame = frame.filter(pl.any_horizontal(pl.exclude("date").is_not_null()))
        variables = tuple(c for c in monthly.columns if c != "date")
        logger.info(
            f"dataset: {frame.height} quarters {frame['date'].min()} -> {frame['date'].max()}, "
            f"{len(variables)} monthly variables, {frame.width - 1 - len(self.targets)} feature columns"
        )
        return QuarterlyDataset(frame=frame, targets=self.targets, variables=variables)

    def build_from_iep(self, data_dir: str = config.IEP_DATA_DIR) -> QuarterlyDataset:
        from data_iep.pipeline import load_iep_data

        data = load_iep_data(data_dir)
        return self.build(data["monthly"], data["quarterly"])
