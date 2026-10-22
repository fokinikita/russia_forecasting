"""Dynamic factor model for all targets and horizons at once.

statsmodels DynamicFactorMQ (EM estimation, handles missing values natively).
Endogenous series = 4 targets + the selected monthly features (month `availability`
of every quarter). Real-time forecast from origin T:

    * parameters are estimated once per refit point on data known then;
    * the model is re-filtered on data up to T, with the targets of T unknown;
    * smoothed value at T = nowcast (h = 1), then h - 1 steps ahead.

Unlike the original DFM pipeline no future feature values are used for h > 1.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from typing import Optional

import attrs
import numpy as np
import pandas as pd
import polars as pl

import config
from forecasting.dataset import QuarterlyDataset, shift_quarters

logger = logging.getLogger(__name__)


def _to_period_frame(frame: pl.DataFrame, columns: list[str]) -> pd.DataFrame:
    pdf = frame.select(["date", *columns]).to_pandas().set_index("date")
    pdf.index = pd.PeriodIndex(pd.to_datetime(pdf.index), freq="Q")
    return pdf.astype(float)


@attrs.define(slots=True)
class DFMForecaster:
    dataset: QuarterlyDataset
    feature_cols: tuple[str, ...]
    train_start_year: int
    factors: int = config.RT_DFM_FACTORS
    factor_order: int = config.RT_DFM_FACTOR_ORDER
    maxiter: int = 300
    results: object = None
    _data: Optional[pd.DataFrame] = None

    name = "dfm"

    @property
    def columns(self) -> list[str]:
        return [*self.dataset.target_columns, *self.feature_cols]

    def _history(self) -> pd.DataFrame:
        if self._data is None:
            frame = self.dataset.frame.filter(pl.col("date").dt.year() >= self.train_start_year)
            data = _to_period_frame(frame, self.columns)
            # series with no data at all would break the EM step
            self._data = data.loc[:, data.notna().sum() >= 8]
        return self._data

    def fit(self, known_until: date) -> "DFMForecaster":
        """Estimate on information available when targets are known up to `known_until`."""
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

        data = self._history().loc[: pd.Period(known_until, "Q")]
        data = data.loc[:, data.notna().sum() >= 8]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DynamicFactorMQ(
                data,
                factors=self.factors,
                factor_orders=self.factor_order,
                idiosyncratic_ar1=True,
                standardize=True,
            )
            self.results = model.fit(disp=False, maxiter=self.maxiter)
        return self

    def forecast(self, origin: date, horizon: int) -> pl.DataFrame:
        """Real-time forecasts from `origin`: target | horizon | target_date | pred."""
        columns = list(self.results.model.endog_names)
        origin_q = pd.Period(origin, "Q")
        data = self._history().reindex(columns=columns).loc[:origin_q].copy()
        if data.index[-1] != origin_q:  # origin row is missing completely -> add it empty
            data.loc[origin_q] = np.nan
        target_cols = [c for c in self.dataset.target_columns if c in columns]
        data.loc[origin_q, target_cols] = np.nan  # targets of the origin quarter are not known yet
        # publication gaps at the origin: carry the last value forward (like the direct models)
        feature_cols = [c for c in columns if c not in target_cols]
        data.loc[origin_q, feature_cols] = data[feature_cols].ffill(limit=1).loc[origin_q]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.results.apply(data)
            pred = applied.get_prediction(
                start=origin_q, end=origin_q + (horizon - 1), information_set="smoothed"
            ).predicted_mean

        rows = []
        for target_col in target_cols:
            target = target_col.removesuffix("_log_d4")
            for h in range(1, horizon + 1):
                rows.append(
                    {
                        "target": target,
                        "horizon": h,
                        "origin": origin,
                        "target_date": shift_quarters(origin, h - 1),
                        "pred": float(pred[target_col].iloc[h - 1]),
                    }
                )
        return pl.DataFrame(rows)
