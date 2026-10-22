"""Collect everything from data.iep.ru, transform it and save it.

    python -m data_iep.pipeline          # from the project root

Writes to config.IEP_DATA_DIR:
    raw_quarterly / raw_monthly / raw_daily .parquet   - as published
    quarterly / monthly / daily .parquet               - transformed, model-ready
    metadata.csv                                       - vintage id, unit, dates per series
"""

from __future__ import annotations

import logging
from pathlib import Path

import attrs
import polars as pl

import config
from data_iep.client import IepClient
from data_iep.collectors import DailyCollector, MonthlyCollector, QuarterlyCollector
from data_iep.transform import IepTransformService

logger = logging.getLogger(__name__)


@attrs.define(slots=True)
class IepDataPipeline:
    output_dir: Path = attrs.field(default=Path(config.IEP_DATA_DIR), converter=Path)
    client: IepClient = attrs.field(factory=IepClient)
    transformer: IepTransformService = attrs.field(factory=IepTransformService)
    strict: bool = False
    errors: dict[str, str] = attrs.field(factory=dict)

    def collect(self) -> dict[str, pl.DataFrame]:
        collectors = {
            "quarterly": QuarterlyCollector(client=self.client, strict=self.strict),
            "monthly": MonthlyCollector(client=self.client, strict=self.strict),
            "daily": DailyCollector(client=self.client, strict=self.strict),
        }
        raw = {name: collector.collect() for name, collector in collectors.items()}
        raw["metadata"] = pl.concat([c.metadata_frame() for c in collectors.values()], how="diagonal")
        self.errors = {k: v for c in collectors.values() for k, v in c.errors.items()}
        if self.errors:
            logger.error(f"{len(self.errors)} series failed: {list(self.errors)}")
        return raw

    def run(self, save: bool = True) -> dict[str, pl.DataFrame]:
        raw = self.collect()
        quarterly, monthly, daily = self.transformer.run(raw["quarterly"], raw["monthly"], raw["daily"])
        result = {
            "raw_quarterly": raw["quarterly"],
            "raw_monthly": raw["monthly"],
            "raw_daily": raw["daily"],
            "quarterly": quarterly,
            "monthly": monthly,
            "daily": daily,
            "metadata": raw["metadata"],
        }
        if save:
            self.save(result)
        return result

    def save(self, result: dict[str, pl.DataFrame]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in result.items():
            if name == "metadata":
                frame.write_csv(self.output_dir / "metadata.csv")
            else:
                frame.write_parquet(self.output_dir / f"{name}.parquet")
        logger.info(f"Saved IEP data to {self.output_dir.resolve()}")


def load_iep_data(output_dir: str | Path = config.IEP_DATA_DIR) -> dict[str, pl.DataFrame]:
    """Read the transformed datasets saved by IepDataPipeline (no network)."""
    output_dir = Path(output_dir)
    return {name: pl.read_parquet(output_dir / f"{name}.parquet") for name in ("quarterly", "monthly", "daily")}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    data = IepDataPipeline().run()
    for name in ("quarterly", "monthly", "daily"):
        frame = data[name]
        print(f"{name:9s} {frame.shape}  {frame['date'].min()} -> {frame['date'].max()}")
