import logging
import os
import warnings

import attrs
import polars as pl
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

import config
from models.dfm import DFM
from preprocess_data.datae2e import DataE2E

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


def run_main_dfm() -> None:
    train, valid, train_valid, test, avail_features_full = DataE2E().run()
    features = pl.concat([train, valid, test])

    targets = ["gdp_log_d4", "cons_log_d4", "inv_log_d4", "inv_cap_log_d4"]

    forecasts_steps = range(config.TEST_LEN + config.HORIZON - 1)

    features_type_grid = ["d12"]
    features_strategy_grid = ["avail_only", "all"]
    avaliability_grid = range(1, config.MAX_AVALIABILITY + 1)
    k_factors_grid = [1, 2, 3]
    factor_order_grid = [
        1,
        2,
    ]

    pred_dfm_list = []

    for forecast_length in forecasts_steps:

        end_date_dfm = features["date"].max() - relativedelta(
            months=3 * (config.TEST_LEN - forecast_length - 1)
        )
        start_date_dfm = features["date"].max() - relativedelta(
            months=3 * (config.TEST_LEN - 1 - forecast_length - 1 + config.HORIZON)
        )
        end_date_dfm, start_date_dfm

        train_dfm = features.filter((pl.col("date") < start_date_dfm))
        test_dfm = features.filter(
            (pl.col("date") >= start_date_dfm) & (pl.col("date") <= end_date_dfm)
        )

        test_dates = sorted(test_dfm["date"].unique().to_list())
        logger.info(f"Start training DFM to predict {test_dates} period")

        for features_type in features_type_grid:
            for features_strategy in features_strategy_grid:
                for avaliability in avaliability_grid:
                    for k_factors in k_factors_grid:
                        for factor_order in factor_order_grid:

                            logger.info(
                                f"DFM has parameters features_type={features_type}, "
                                f"features_strategy={features_strategy}, avaliability={avaliability}"
                            )
                            logger.info(
                                f"k_factors={k_factors}, factor_order={factor_order}"
                            )
                            dfm = DFM(
                                train_dfm=train_dfm,
                                targets=targets,
                                avaliability=avaliability,
                                k_factors=k_factors,
                                factor_order=factor_order,
                                avail_features_full=avail_features_full,
                                features_type=features_type,
                                features_strategy=features_strategy,
                            )

                            dfm.fit()

                            pred_dfm = dfm.predict(test_dfm)

                            pred_dfm = pred_dfm.with_columns(
                                pl.lit(avaliability).alias("avaliability"),
                                pl.lit(features_type).alias("features_type"),
                                pl.lit(features_strategy).alias("features_strategy"),
                                pl.lit(k_factors).alias("k_factors"),
                                pl.lit(factor_order).alias("factor_order"),
                            )

                            pred_dfm_list.append(pred_dfm)
                            logger.info(f"Successfully fitted")

    dfm_preds_pl_df = pl.concat(pred_dfm_list)
    os.makedirs("preds", exist_ok=True)
    dfm_preds_pl_df.write_csv("preds/dfm_pred_test.csv")


if __name__ == "__main__":
    run_main_dfm()
