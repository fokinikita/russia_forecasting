import attrs
import polars as pl
import config


@attrs.define(slots=True)
class LagsService:
    features: pl.DataFrame

    def _dict_to_list(self, features_dict: dict) -> list:
        if isinstance(features_dict, dict):
            return [val for value in features_dict.values() for val in self._dict_to_list(value)]
        elif isinstance(features_dict, (list, tuple, set)):
            return [val for value in features_dict for val in self._dict_to_list(value)]
        else:
            return [features_dict]
    def _get_monthly_lags(self, features_dict: dict):
        columns_list = self._dict_to_list(features_dict)

        lags_expr = []
        for horizon in range(1, config.HORIZON):
            for col in columns_list:
                lags_expr.append(
                    pl.col(col).shift(horizon).alias(col + f"_lag{horizon}")
                )

        self.features = self.features.with_columns(
            lags_expr
        )

    def _get_quarterly_lags(self, columns_d4: list[str]):

        lags_expr=[]
        for horizon in range(1, config.HORIZON + 1):
            for col in columns_d4:
                lags_expr.append(
                    pl.col(col).shift(horizon).alias(col + f"_lag{horizon}")
                )

        self.features = self.features.with_columns(
            lags_expr
        )

    def get_lags(self, columns_d4: list[str], features_dict: dict) -> pl.DataFrame:
        self._get_monthly_lags(features_dict)
        self._get_quarterly_lags(columns_d4)

        return self.features