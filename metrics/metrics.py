from typing import Optional

import attrs
import polars as pl
from dateutil.relativedelta import relativedelta

import config


@attrs.define(slots=True)
class MetricsCalculator:
    preds_list: list[pl.DataFrame]
    model_names: list[str]
    target_names: list[str]
    features: Optional[pl.DataFrame] = None

    def _concat_models(self):

        all_targets_preds = []
        for target in self.target_names:
            for model_index, model_name in enumerate(self.model_names):
                single_target_pred = (
                    self.preds_list[model_index]
                    .filter(pl.col("target_name") == target)
                    .join(
                        self.features.select(["date", target]).rename(
                            {target: "target_value"}
                        ),
                        on="date",
                    )
                    .with_columns(
                        (
                            (pl.col(f"pred_{model_name}") - pl.col("target_value")) ** 2
                        ).alias("mse")
                    )
                )

                single_target_pred_gb = (
                    single_target_pred.group_by(
                        ["horizon", "avaliability", "features_type"]
                    )
                    .agg((pl.col("mse").mean() ** 0.5).alias(f"rmse"))
                    .sort(["horizon", "avaliability", "features_type"])
                    .with_columns(
                        pl.lit(target).alias("target_name"),
                        pl.lit(model_name).alias("model"),
                    )
                )
                all_targets_preds.append(single_target_pred_gb)

        return pl.concat(all_targets_preds)

    def _get_naive(self):
        targets = self.features.select(["date"] + self.target_names).sort("date")

        test_end_date = self.features["date"].max()
        test_start_date = test_end_date - relativedelta(
            months=3 * (config.TEST_LEN - 1)
        )

        pred_dates = []
        naive_list = []
        for horizon in range(1, config.HORIZON + 1):
            for target in self.target_names:
                naive_one_horizon_target = (
                    targets.with_columns(
                        pl.col(target).shift(horizon).alias("pred_naive"),
                        pl.lit(target).alias("target_name"),
                        pl.lit(horizon).alias("horizon"),
                    )
                    .with_columns(
                        ((pl.col(target) - pl.col("pred_naive")) ** 2).alias("mse")
                    )
                    .filter(pl.col("date").is_between(test_start_date, test_end_date))
                )

                naive_one_horizon_target_gb = naive_one_horizon_target.group_by(
                    "target_name", "horizon"
                ).agg((pl.col("mse").mean() ** 0.5).alias("rmse_naive"))

                naive_list.append(naive_one_horizon_target_gb)

        return pl.concat(naive_list)

    def get_metrics(self):
        naive_pred = self._get_naive()
        models_pred = self._concat_models()

        all_pred = models_pred.join(
            naive_pred, on=["horizon", "target_name"], how="left"
        ).with_columns((pl.col("rmse") / pl.col("rmse_naive")).alias("rel_rmse_naive"))

        return all_pred

def run_mfbvar_metrics_calculator(test: pl.DataFrame) -> pl.DataFrame:
    mfbvar_pred_pl = pl.read_csv('preds/mfbvar_pred_test.csv').with_columns(
        pl.col('fcst_date').cast(pl.Date).alias('date')
    ).with_columns(
        (pl.col('date').dt.offset_by("-2mo")).alias('date')
    ).rename(
        {
            'median': 'pred_mfbvar',
            'variable': 'target_name'
        }
    ).drop('fcst_date').drop('p')

    fact_melted = pl.concat(
        [test.select(['date', target]).rename(
            {
                target: 'fact'
            }
        ).with_columns(
            pl.lit(target).alias('target_name')
        ) for target in target_names
        ]
    )

    mfbvar_pred_with_fact = mfbvar_pred_pl.filter(
        pl.col('date').is_in(test['date'].unique().to_list())
    ).join(fact_melted, on=['date', 'target_name']).with_columns(
        ((pl.col('fact') - pl.col('pred_mfbvar')) ** 2).alias('mse')
    )

    mfbvar_pred_metrics = mfbvar_pred_with_fact.group_by(
        'horizon',
        'avaliability',
        'target_name',
    ).agg(
        (pl.col('mse').mean() ** 0.5).alias('rmse')
    )
    mfbvar_pred_metrics = mfbvar_pred_metrics.join(
        metrics_naive, on=['horizon', 'target_name']
    ).with_columns(
        (pl.col('rmse') / pl.col('rmse_naive')).alias('rel_rmse_naive')
    )

    mfbvar_pred_metrics = mfbvar_pred_metrics.sort('avaliability', descending=True).sort('horizon').sort('target_name')

    mfbvar_pred_metrics = mfbvar_pred_metrics.with_columns(
        pl.lit('mfbvar').alias('model'),
        pl.lit('d12').alias('features_type'),
    )

    metrics_mfbvar = mfbvar_pred_metrics.select(
        [
            'horizon',
             'avaliability',
             'features_type',
             'rmse',
             'target_name',
             'model',
             'rmse_naive',
             'rel_rmse_naive'
        ]
    )

    return metrics_mfbvar