import attrs
import polars as pl

from preprocess_data.lags_service import LagsService
from preprocess_data.montlhy_to_quarterly import MonthlyToQuarterlyService
from preprocess_data.prepare_data import FeaturesService
from preprocess_data.splitter_service import TrainValTestSplit


@attrs.define(slots=True)
class DataE2E:
    @staticmethod
    def run() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:

        features_service = FeaturesService()
        monthly_data, quarterly_data = features_service.get_features()

        columns_d12 = features_service.columns_d12
        columns_rolling = features_service.columns_rolling
        columns_d4 = features_service.columns_d4

        mtoq_service = MonthlyToQuarterlyService(columns_d12, columns_rolling)
        features = mtoq_service.run_transorm(monthly_data, quarterly_data)

        avail_features_full = mtoq_service.avail_features_full

        lags_service = LagsService(features)
        features = lags_service.get_lags(columns_d4, avail_features_full)

        splitter_service = TrainValTestSplit(features)
        train, valid, test = splitter_service.split()
        train_valid = pl.concat([train, valid])

        return (train, valid, train_valid, test, avail_features_full)
