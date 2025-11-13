import attrs
import polars as pl
import pandas as pd

from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

import config

@attrs.define(slots=True)
class DFM:
    train_dfm: pl.DataFrame
    targets: list[str]
    avaliability: int
    k_factors: int
    factor_order: int
    avail_features_full: dict
    features_type: str = 'd12'
    features_strategy: str = 'avail_only'
    model_fitted: object = None

    @staticmethod
    def _filter_nan_features(
        train: pl.DataFrame,
        features_cols: list[str]
    ) -> list[str]:
        df_train = train.select(features_cols).to_pandas()
        mask = ~df_train.isna().any()

        return [col for col, ok in mask.items() if ok]

    def _get_features_names(self) -> list[str]:
        if self.features_type == 'd12':
            if self.features_strategy == 'avail_only':
                if self.avaliability == 1:
                    feature_cols = self.avail_features_full[self.features_type][1]
                elif self.avaliability == 2:
                    feature_cols = self.avail_features_full[self.features_type][2]
                elif self.avaliability == 3:
                    feature_cols = self.avail_features_full[self.features_type][3]
                else:
                    raise ValueError("avaliability must be between 1 and 3")
            elif self.features_strategy == 'all':
                if self.avaliability == 1:
                    feature_cols = self.avail_features_full[self.features_type][1]
                elif self.avaliability == 2:
                    feature_cols = (self.avail_features_full[self.features_type][1]
                                    + self.avail_features_full[self.features_type][2])
                elif self.avaliability == 3:
                    feature_cols = (self.avail_features_full[self.features_type][1] +
                                    self.avail_features_full[self.features_type][2] +
                                    self.avail_features_full[self.features_type][3])
                else:
                    raise ValueError("avaliability must be between 1 and 3")
            else:
                raise ValueError("features_strategy must be 'avail_only' or 'all'")
        elif self.features_type == 'rolling':
            if self.avaliability == 1:
                feature_cols = self.avail_features_full[self.features_type][1]
            elif self.avaliability == 2:
                feature_cols = self.avail_features_full[self.features_type][2]
            elif self.avaliability == 3:
                feature_cols = self.avail_features_full[self.features_type][3]
            else:
                raise ValueError("avaliability must be between 1 and 3")
        else:
            raise ValueError("features_typey must be 'd12' (default) or 'rolling'")

        feature_cols_no_nans = self._filter_nan_features(self.train_dfm, feature_cols)

        return feature_cols_no_nans

    def fit(self):

        feature_cols_no_nans = self._get_features_names()

        endog = self.train_dfm.select(['date'] + self.targets).to_pandas().set_index('date')
        exog = self.train_dfm.select(['date'] + feature_cols_no_nans).to_pandas().set_index('date')

        model = DynamicFactor(
            endog=endog,
            exog=exog,
            k_factors=self.k_factors,
            factor_order=self.factor_order,
        )

        self.model_fitted = model.fit(disp=False, maxiter=5000, method="nm")

    def predict(self, test_dfm: pl.DataFrame) -> pl.DataFrame:

        feature_cols_no_nans = self._get_features_names()

        exog_pred = test_dfm.select(['date'] + feature_cols_no_nans).to_pandas().set_index('date')

        forecast = self.model_fitted.forecast(
            steps=len(exog_pred),
            exog=exog_pred
        )

        forecast_pl = pl.DataFrame(forecast.reset_index().rename(columns={'index': 'date'}))
        forecast_pl = forecast_pl.with_columns(
            pl.col('date').cast(pl.Date),
        ).with_columns(
             pl.Series("horizon", range(1, forecast_pl.height + 1))
        )
        forecast_pl = forecast_pl.rename(
            {
                target: target + '_dfm_pred' for target in self.targets
            }
        )
        return forecast_pl
