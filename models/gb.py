import attrs
import polars as pl
from catboost import CatBoostRegressor, Pool

from typing import Optional
import config

@attrs.define(slots=True)
class GB:
    features_type: str
    avaliability: int
    target_name: str
    horizon: int
    avail_features_full: dict
    params: Optional[dict] = {}
    model: Optional[CatBoostRegressor] = None

    @staticmethod
    def _get_actual_lags_features(features_list: list[str], horizon: int) -> list[str]:
        if horizon > 1:
            features_list_actual_lags = []
            for feature in features_list:
                features_list_actual_lags.append(feature + f"_lag{horizon - 1}")

            return features_list_actual_lags

        elif horizon == 1:
            return features_list
        else:
            raise ValueError(f"Invalid horizon: {horizon}, must be >= 1")

    @staticmethod
    def _get_lags_target(target_name, horizon: int) -> list[str]:
        target_lags = target_name + f'_lag{horizon}'

        return [target_lags]


    def fit(
            self,
            train: pl.DataFrame,
            valid: Optional[pl.DataFrame] = None,
            early_stopping: Optional[int] = None
    ):

        base_params = {
            "random_seed": config.RANDOM_SEED,
            "loss_function": "RMSE",
            "verbose": False,
            "thread_count": config.THREAD_COUNT
        }

        if self.params:
            base_params.update(**self.params)

        features_cols = self._get_actual_lags_features(
            self.avail_features_full[self.features_type][self.avaliability],
            self.horizon
        ) + self._get_lags_target(self.target_name, self.horizon)

        target_col = self.target_name

        train_pool = Pool(
            data=train.select(features_cols).to_pandas(),
            label=train.select(target_col).to_pandas(),
        )

        if valid is not None and not valid.is_empty():
            valid_pool = Pool(
                data=valid.select(features_cols).to_pandas(),
                label=valid.select(target_col).to_pandas(),
            )

            base_params["early_stopping_rounds"] = early_stopping
            self.model = CatBoostRegressor(**base_params, eval_metric=base_params["loss_function"])
            self.model.fit(train_pool, eval_set=valid_pool)

        else:
            if hasattr(self.model, "best_iteration_") and self.model.best_iteration_ is not None:
                base_params["iterations"] = self.model.best_iteration_ + 1

            self.model = CatBoostRegressor(**base_params)
            self.model.fit(train_pool)

    def predict(self, test: pl.DataFrame):

        features_cols = self._get_actual_lags_features(
            self.avail_features_full[self.features_type][self.avaliability],
            self.horizon
        ) + self._get_lags_target(self.target_name, self.horizon)

        test_pool = Pool(data=test.select(features_cols).to_pandas())

        pred_df = test.select('date').with_columns(
            pl.lit(self.model.predict(test_pool)).alias('pred_gb'),
            pl.lit(self.horizon).alias("horizon"),
            pl.lit(self.avaliability).alias("avaliability"),
            pl.lit(self.target_name).alias('target_name'),
        )
        return pred_df
