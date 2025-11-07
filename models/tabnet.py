import attrs
import numpy as np
import polars as pl
from typing import Optional, Dict, Any
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler


@attrs.define(slots=True)
class TabNetModel:
    features_type: str
    avaliability: int
    target_name: str
    horizon: int
    avail_features_full: dict
    params: Dict[str, Any] = attrs.field(factory=dict)
    model: Optional[TabNetRegressor] = None
    scaler: Optional[StandardScaler] = None
    feature_cols_no_nans: Optional[list[str]] = None

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

    @staticmethod
    def _prepare_numpy(df: pl.DataFrame, features_cols: list[str], target_name: str) -> tuple[np.ndarray, np.ndarray]:
        df_pd = df.drop("date").to_pandas()
        X = df_pd[features_cols].astype(np.float32).values
        y = df_pd[target_name].astype(np.float32).values.reshape(-1, 1)
        return X, y

    @staticmethod
    def _filter_nan_features(
        train: pl.DataFrame,
        valid: Optional[pl.DataFrame],
        features_cols: list[str]
    ) -> list[str]:
        df_train = train.select(features_cols).to_pandas()
        valid_pd = valid.select(features_cols).to_pandas() if valid is not None and not valid.is_empty() else None

        mask = ~df_train.isna().any()
        if valid_pd is not None:
            mask &= ~valid_pd.isna().any()

        return [col for col, ok in mask.items() if ok]

    def fit(
        self,
        train: pl.DataFrame,
        valid: Optional[pl.DataFrame] = None,
    ):

        features_cols = (
            self._get_actual_lags_features(
                self.avail_features_full[self.features_type][self.avaliability],
                self.horizon
            ) + self._get_lags_target(self.target_name, self.horizon)
        )

        self.feature_cols_no_nans = self._filter_nan_features(train, valid, features_cols)

        self.scaler = StandardScaler()
        X_train, y_train = self._prepare_numpy(train, self.feature_cols_no_nans, self.target_name)
        X_train = self.scaler.fit_transform(X_train)

        tabnet_params = dict(
            optimizer_params={"lr": self.params.get("learning_rate", 0.05)},
            mask_type=self.params.get("mask_type", "sparsemax"),
            lambda_sparse=self.params.get("lambda_sparse", 0.005),
            device_name=self.params.get("device_name", "cpu"),
            verbose=0,
        )

        if valid is not None and not valid.is_empty():
            X_valid, y_valid = self._prepare_numpy(valid, self.feature_cols_no_nans, self.target_name)
            X_valid = self.scaler.transform(X_valid)

            self.model = TabNetRegressor(**tabnet_params)

            self.model.fit(
                X_train=X_train, y_train=y_train,
                eval_set=[(X_valid, y_valid)],
                eval_name=["valid"],
                eval_metric=["rmse"],
                max_epochs=self.params.get("max_epochs", 200),
                patience=self.params.get("early_stopping_rounds", 20),
                batch_size=self.params.get("batch_size", 8),
                virtual_batch_size=self.params.get("virtual_batch_size", 8),
                drop_last=False,
            )

            best_epoch = np.argmin(self.model.history["valid_rmse"]) + 1
            if best_epoch < 10:
                self.params["max_epochs"] = 10
            else:
                self.params["max_epochs"] = np.argmin(self.model.history["valid_rmse"]) + 1

        else:
            self.model = TabNetRegressor(**tabnet_params)
            self.model.fit(
                X_train=X_train, y_train=y_train,
                max_epochs=self.params.get("max_epochs", 200),
                batch_size=self.params.get("batch_size", 8),
                virtual_batch_size=self.params.get("virtual_batch_size", 8),
                drop_last=False,
            )

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        X_test, _ = self._prepare_numpy(test, self.feature_cols_no_nans, self.target_name)
        X_test = self.scaler.transform(X_test)
        preds = self.model.predict(X_test).flatten()

        pred_df = test.select("date").with_columns(
            pl.Series(preds).alias("pred_tabnet"),
            pl.lit(self.horizon).alias("horizon"),
            pl.lit(self.avaliability).alias("avaliability"),
            pl.lit(self.target_name).alias("target_name"),
            pl.lit(self.features_type).alias("features_type"),
        )
        return pred_df
