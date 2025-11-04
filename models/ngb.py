import attrs
import numpy as np
import polars as pl
from typing import Optional, Dict, Any
from ngboost import NGBoost
from ngboost.distns import Normal
from ngboost.scores import CRPS
from sklearn.tree import DecisionTreeRegressor

import config

@attrs.define(slots=True)
class NGB:
    features_type: str
    avaliability: int
    target_name: str
    horizon: int
    avail_features_full: dict
    params: Dict[str, Any] = attrs.field(factory=dict)
    model: Optional[NGBoost] = None

    @staticmethod
    def _get_actual_lags_features(features_list: list[str], horizon: int) -> list[str]:
        if horizon > 1:
            return [f"{feature}_lag{horizon - 1}" for feature in features_list]
        elif horizon == 1:
            return features_list
        else:
            raise ValueError(f"Invalid horizon: {horizon}, must be >= 1")

    @staticmethod
    def _get_lags_target(target_name: str, horizon: int) -> list[str]:
        return [f"{target_name}_lag{horizon}"]

    def _prepare_numpy(self, df: pl.DataFrame, features_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        df_pd = df.drop('date').to_pandas()
        X = df_pd[features_cols].astype(np.float32).values
        y = df_pd[self.target_name].astype(np.float32).values
        return X, y

    def fit(
        self,
        train: pl.DataFrame,
        valid: Optional[pl.DataFrame] = None,
        early_stopping: Optional[int] = 30,
    ):

        features_cols = (
            self._get_actual_lags_features(
                self.avail_features_full[self.features_type][self.avaliability],
                self.horizon
            )
            + self._get_lags_target(self.target_name, self.horizon)
        )

        X_train, y_train = self._prepare_numpy(train, features_cols)

        if valid is not None and not valid.is_empty():
            X_valid, y_valid = self._prepare_numpy(valid, features_cols)

            base_params = {
                "Base": DecisionTreeRegressor(
                    criterion="friedman_mse",
                    max_depth=config.DEPTH_NGB_BASE,
                    random_state=self.params.get("random_state", config.RANDOM_SEED),
                ),
                "Dist": Normal,
                "Score": CRPS,
                "n_estimators": self.params.get("n_estimators", config.START_ITERATIONS_NGB),
                "learning_rate": self.params.get("learning_rate", config.LEARNING_RATE_NGB),
                "verbose": self.params.get("verbose", True),
                "random_state": self.params.get("random_state", config.RANDOM_SEED),
                "early_stopping_rounds": early_stopping,
                "validation_fraction": None,
                "verbose": False
            }

            self.model = NGBoost(**base_params)
            self.model.fit(X_train, y_train, X_val=X_valid, Y_val=y_valid)

            best_n_estimators = self.model.best_val_loss_itr + 1
            self.params["n_estimators"] = best_n_estimators

        else:
            best_n_estimators = self.params.get("n_estimators", config.START_ITERATIONS_NGB)

        if valid is not None and not valid.is_empty():
            train_valid = pl.concat([train, valid])
        else:
            train_valid = train

        X_train_valid, y_train_valid = self._prepare_numpy(train_valid, features_cols)

        final_model = NGBoost(
            Base=DecisionTreeRegressor(
                criterion="friedman_mse",
                max_depth=config.DEPTH_NGB_BASE,
                random_state=self.params.get("random_state", config.RANDOM_SEED),
            ),
            Dist=Normal,
            Score=CRPS,
            n_estimators=best_n_estimators,
            learning_rate=self.params.get("learning_rate", config.LEARNING_RATE_NGB),
            verbose=self.params.get("verbose", True),
            random_state=self.params.get("random_state", config.RANDOM_SEED),
        )

        final_model.fit(X_train_valid, y_train_valid)
        self.model = final_model

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:

        features_cols = (
            self._get_actual_lags_features(
                self.avail_features_full[self.features_type][self.avaliability],
                self.horizon
            )
            + self._get_lags_target(self.target_name, self.horizon)
        )

        X_test, _ = self._prepare_numpy(test, features_cols)
        preds = self.model.predict(X_test)

        pred_df = test.select("date").with_columns(
            pl.Series(preds.ravel()).alias("pred_ngb"),
            pl.lit(self.horizon).alias("horizon"),
            pl.lit(self.avaliability).alias("avaliability"),
            pl.lit(self.target_name).alias("target_name"),
        )

        return pred_df
