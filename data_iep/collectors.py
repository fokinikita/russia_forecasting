"""Collectors: download raw series from data.iep.ru into one wide DataFrame per frequency.

    QuarterlyCollector().collect() -> date | gdp_2021 | gdp_2008 | ... (quarter start dates)
    MonthlyCollector().collect()   -> date | ipi | cpi | ...           (month start dates)
    DailyCollector().collect()     -> date | miacr                     (calendar dates)

Values are returned exactly as published (chained %, levels, ...).
Turning them into model-ready "basic" series is the job of transform.py.
"""

from __future__ import annotations

import logging

import attrs
import polars as pl

from data_iep.client import IepClient
from data_iep.series_config import (
    DAILY_SERIES,
    MONTHLY_SERIES,
    QUARTERLY_SERIES,
    Frequency,
    SeriesSpec,
)

logger = logging.getLogger(__name__)

_IEP_FREQUENCY = {"Q": "quarterly", "M": "monthly", "D": "daily"}
_TRUNCATE = {"Q": "1q", "M": "1mo", "D": "1d"}


class SeriesNotFoundError(LookupError):
    pass


def find_slice(slices: list[dict], spec: SeriesSpec) -> dict:
    """Pick the single line of a dataset that matches spec.dims / unit / frequency."""
    candidates = []
    for line in slices:
        selector = line["selector"]
        values = {v for k, v in selector.items() if k.startswith("p")}
        if spec.unit_id is not None and selector.get("unit_id") != spec.unit_id:
            continue
        if line.get("frequency") != _IEP_FREQUENCY[spec.frequency]:
            continue
        if all(dim in values for dim in spec.dims):
            candidates.append(line)

    if len(candidates) != 1:
        available = "\n  ".join(line["dimension"] for line in slices)
        raise SeriesNotFoundError(
            f"{spec.name}: expected exactly one line, found {len(candidates)} "
            f"for dims={spec.dims} unit_id={spec.unit_id}. Available lines:\n  {available}"
        )
    return candidates[0]


@attrs.define(slots=True)
class BaseCollector:
    frequency: Frequency
    specs: list[SeriesSpec]
    client: IepClient = attrs.field(factory=IepClient)
    strict: bool = False  # True -> raise on the first missing series
    metadata: list[dict] = attrs.field(factory=list)
    errors: dict[str, str] = attrs.field(factory=dict)

    def fetch_series(self, spec: SeriesSpec) -> pl.DataFrame:
        vintage = self.client.get_latest_vintage(spec.dataset_uuid, spec.search_query)
        line = find_slice(self.client.get_slices(vintage["vintage_id"]), spec)
        frame = self.client.get_line_frame(line["selector"])

        frame = (
            frame.with_columns(pl.col("date").dt.truncate(_TRUNCATE[self.frequency]))
            # a few datasets have duplicated periods; keep the last published value
            .unique(subset="date", keep="last")
            .sort("date")
            .rename({"value": spec.name})
        )

        self.metadata.append(
            {
                "name": spec.name,
                "frequency": self.frequency,
                "kind": spec.kind,
                "dataset_uuid": spec.dataset_uuid,
                "dataset_name": vintage.get("dataset_name"),
                "vintage_id": vintage["vintage_id"],
                "vintage_date": vintage.get("vintage_date"),
                "dimension": line["dimension"],
                "unit": (line.get("unit") or {}).get("name"),
                "n_obs": frame.height,
                "first_date": frame["date"].min() if frame.height else None,
                "last_date": frame["date"].max() if frame.height else None,
            }
        )
        return frame

    def collect(self) -> pl.DataFrame:
        self.metadata, self.errors = [], {}
        frames = []
        for spec in self.specs:
            try:
                frames.append(self.fetch_series(spec))
                logger.info(f"[{self.frequency}] {spec.name}: ok")
            except Exception as error:  # one broken series must not stop the run
                if self.strict:
                    raise
                self.errors[spec.name] = str(error)
                logger.error(f"[{self.frequency}] {spec.name}: {error}")

        if not frames:
            return pl.DataFrame(schema={"date": pl.Date})

        wide = frames[0]
        for frame in frames[1:]:
            wide = wide.join(frame, on="date", how="full", coalesce=True)
        return wide.sort("date")

    def metadata_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.metadata)


def QuarterlyCollector(**kwargs) -> BaseCollector:
    return BaseCollector(frequency="Q", specs=QUARTERLY_SERIES, **kwargs)


def MonthlyCollector(**kwargs) -> BaseCollector:
    return BaseCollector(frequency="M", specs=MONTHLY_SERIES, **kwargs)


def DailyCollector(**kwargs) -> BaseCollector:
    return BaseCollector(frequency="D", specs=DAILY_SERIES, **kwargs)
