"""Model wrappers with one interface: fit(X, y) -> predict(X).

Training protocol (same as the original project, per model):
    1. the last `valid_len` rows of the training sample are a validation set,
       the model is fitted with early stopping on it;
    2. the model is re-fitted on the whole sample with the best number of
       iterations / epochs found in step 1.

DFM is not a direct model (one state-space model for all targets and horizons),
it lives in forecasting/dfm.py.
"""

from __future__ import annotations

import abc
import logging
import warnings
from typing import ClassVar, Optional

import attrs
import numpy as np

import config

logger = logging.getLogger(__name__)


@attrs.define(slots=True)
class MissingValuePolicy:
    """For models that cannot handle NaN: drop sparse columns, fill the rest with train medians."""

    max_missing_share: float = 0.3
    keep: Optional[np.ndarray] = None
    fill: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "MissingValuePolicy":
        missing_share = np.isnan(x).mean(axis=0)
        self.keep = missing_share <= self.max_missing_share
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(x[:, self.keep], axis=0)
        self.fill = np.where(np.isnan(medians), 0.0, medians)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x[:, self.keep].copy()
        rows, cols = np.where(np.isnan(x))
        x[rows, cols] = self.fill[cols]
        return x


@attrs.define(slots=True)
class DirectForecaster(abc.ABC):
    """Base class: early stopping on the tail of the sample, then refit on all of it."""

    name: ClassVar[str] = "base"
    valid_len: int = config.RT_VALID_LEN
    seed: int = config.RANDOM_SEED
    min_train_rows: int = 12

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DirectForecaster":
        n_valid = self.valid_len if len(y) - self.valid_len >= self.min_train_rows else 0
        if n_valid:
            best = self._fit_early_stopping(x[:-n_valid], y[:-n_valid], x[-n_valid:], y[-n_valid:])
        else:
            best = None  # sample too short for validation -> default size
        self._fit_final(x, y, best)
        return self

    @abc.abstractmethod
    def _fit_early_stopping(self, x_train, y_train, x_valid, y_valid) -> int:
        """Fit with early stopping, return the best number of iterations / epochs."""

    @abc.abstractmethod
    def _fit_final(self, x: np.ndarray, y: np.ndarray, best: Optional[int]) -> None:
        """Fit on the whole sample with `best` iterations (None -> default)."""

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...


@attrs.define(slots=True)
class CatBoostForecaster(DirectForecaster):
    name: ClassVar[str] = "gb"
    iterations: int = config.START_ITERATIONS
    early_stopping_rounds: int = config.EARLY_STOPPING_ROUNDS
    model: object = None

    def _make(self, iterations: int, **extra):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=iterations,
            loss_function="RMSE",
            random_seed=self.seed,
            thread_count=1,  # parallelism is done across models
            verbose=False,
            allow_writing_files=False,
            **extra,
        )

    def _fit_early_stopping(self, x_train, y_train, x_valid, y_valid) -> int:
        model = self._make(self.iterations, early_stopping_rounds=self.early_stopping_rounds)
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        return max(model.get_best_iteration() + 1, 10)

    def _fit_final(self, x, y, best) -> None:
        self.model = self._make(best or self.iterations)
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


@attrs.define(slots=True)
class NGBoostForecaster(DirectForecaster):
    name: ClassVar[str] = "ngb"
    n_estimators: int = config.START_ITERATIONS_NGB
    learning_rate: float = config.LEARNING_RATE_NGB
    depth: int = config.DEPTH_NGB_BASE
    early_stopping_rounds: int = config.EARLY_STOPPING_ROUNDS_NGB
    policy: MissingValuePolicy = attrs.field(factory=MissingValuePolicy)
    model: object = None

    def _make(self, n_estimators: int, **extra):
        from ngboost import NGBoost
        from ngboost.distns import Normal
        from ngboost.scores import CRPS
        from sklearn.tree import DecisionTreeRegressor

        return NGBoost(
            Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=self.depth, random_state=self.seed),
            Dist=Normal,
            Score=CRPS,
            n_estimators=n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            verbose=False,
            **extra,
        )

    def _fit_early_stopping(self, x_train, y_train, x_valid, y_valid) -> int:
        policy = MissingValuePolicy().fit(x_train)
        model = self._make(
            self.n_estimators, early_stopping_rounds=self.early_stopping_rounds, validation_fraction=None
        )
        model.fit(policy.transform(x_train), y_train, X_val=policy.transform(x_valid), Y_val=y_valid)
        best = model.best_val_loss_itr
        return max((best if best is not None else self.n_estimators - 1) + 1, 10)

    def _fit_final(self, x, y, best) -> None:
        self.policy = MissingValuePolicy().fit(x)
        self.model = self._make(best or self.n_estimators)
        self.model.fit(self.policy.transform(x), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(self.policy.transform(x))


@attrs.define(slots=True)
class TabNetForecaster(DirectForecaster):
    name: ClassVar[str] = "tabnet"
    max_epochs: int = config.MAX_EPOCHS_TABNET
    patience: int = config.EARLY_STOPPING_ROUNDS_TABNET
    batch_size: int = config.BATCH_SIZE_TABNET
    virtual_batch_size: int = config.VIRTUAL_BATCH_SIZE_TABNET
    learning_rate: float = config.LEARNING_RATE_TABNET
    lambda_sparse: float = config.LAMBDA_SPARSE_TABNET
    mask_type: str = config.MASK_TYPE_TABNET
    policy: MissingValuePolicy = attrs.field(factory=MissingValuePolicy)
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    model: object = None

    def _make(self):
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor

        torch.set_num_threads(1)
        return TabNetRegressor(
            optimizer_params={"lr": self.learning_rate},
            mask_type=self.mask_type,
            lambda_sparse=self.lambda_sparse,
            device_name="cpu",
            seed=self.seed,
            verbose=0,
        )

    def _prepare(self, x: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.policy = MissingValuePolicy().fit(x)
        x = self.policy.transform(x)
        if fit:
            self.mean, self.std = x.mean(axis=0), x.std(axis=0)
            self.std[self.std == 0] = 1.0
        return ((x - self.mean) / self.std).astype(np.float32)

    def _fit(self, x, y, max_epochs: int, eval_set=None) -> object:
        model = self._make()
        kwargs = (
            {"eval_set": eval_set, "eval_name": ["valid"], "eval_metric": ["rmse"], "patience": self.patience}
            if eval_set
            else {"patience": 0}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train=x,
                y_train=y.reshape(-1, 1).astype(np.float32),
                max_epochs=max_epochs,
                batch_size=self.batch_size,
                virtual_batch_size=self.virtual_batch_size,
                # a last batch of 1 row breaks batch norm -> drop it only in that case
                drop_last=len(y) % self.batch_size == 1,
                **kwargs,
            )
        return model

    def _fit_early_stopping(self, x_train, y_train, x_valid, y_valid) -> int:
        x_train_s = self._prepare(x_train, fit=True)
        x_valid_s = self._prepare(x_valid, fit=False)
        model = self._fit(
            x_train_s, y_train, self.max_epochs, eval_set=[(x_valid_s, y_valid.reshape(-1, 1).astype(np.float32))]
        )
        return max(int(np.argmin(model.history["valid_rmse"])) + 1, 10)

    def _fit_final(self, x, y, best) -> None:
        self.model = self._fit(self._prepare(x, fit=True), y, best or self.max_epochs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(self._prepare(x, fit=False)).ravel()


DIRECT_MODELS: dict[str, type[DirectForecaster]] = {
    cls.name: cls for cls in (CatBoostForecaster, NGBoostForecaster, TabNetForecaster)
}


def make_direct_model(name: str) -> DirectForecaster:
    try:
        return DIRECT_MODELS[name]()
    except KeyError as error:
        raise ValueError(f"unknown direct model {name!r}, known: {list(DIRECT_MODELS)}") from error
