import attrs
import polars as pl
from dateutil.relativedelta import relativedelta

import config


@attrs.define(slots=True)
class TrainValTestSplit:

    features: pl.DataFrame

    def _filter_start_date(self) -> pl.DataFrame:
        self.features = self.features.filter(
            pl.col("date").dt.year() >= config.START_YEAR
        )
        return self.features

    def split(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        self.features = self._filter_start_date()

        test_end_date = self.features["date"].max()
        test_start_date = test_end_date - relativedelta(
            months=3 * (config.TEST_LEN - 1)
        )

        valid_end_date = test_end_date - relativedelta(months=3 * config.TEST_LEN)
        valid_start_date = valid_end_date - relativedelta(
            months=3 * (config.VALID_LEN - 1)
        )

        test = self.features.filter(
            (pl.col("date") >= test_start_date) & (pl.col("date") <= test_end_date)
        )

        valid = self.features.filter(
            (pl.col("date") >= valid_start_date) & (pl.col("date") <= valid_end_date)
        )

        train = self.features.filter((pl.col("date") < valid_start_date))

        return train, valid, test
