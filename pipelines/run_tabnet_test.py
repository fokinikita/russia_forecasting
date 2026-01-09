import logging
import os

import polars as pl

import config
from models.tabnet import TabNetModel
from preprocess_data.datae2e import DataE2E

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_main_tabnet() -> None:
    horizon_grid = range(1, config.HORIZON + 1)
    features_type_grid = ["rolling", "d12"]
    avaliability_grid = range(1, config.MAX_AVALIABILITY + 1)
    target_grid = ["gdp_log_d4", "cons_log_d4", "inv_log_d4", "inv_cap_log_d4"]

    train, valid, train_valid, test, avail_features_full = DataE2E().run()

    params = {
        "batch_size": config.BATCH_SIZE_TABNET,
        "virtual_batch_size": config.VIRTUAL_BATCH_SIZE_TABNET,
        "learning_rate": config.LEARNING_RATE_TABNET,
        "lambda_sparse": config.LAMBDA_SPARSE_TABNET,
        "mask_type": config.MASK_TYPE_TABNET,
        "max_epochs": config.MAX_EPOCHS_TABNET,
        "early_stopping_rounds": config.EARLY_STOPPING_ROUNDS_TABNET,
        "verbose": False,
    }

    tabnet_pred = []
    for features_type in features_type_grid:
        for target in target_grid:
            for horizon in horizon_grid:
                for avaliability in avaliability_grid:

                    tabnet = TabNetModel(
                        features_type=features_type,
                        avaliability=avaliability,
                        target_name=target,
                        horizon=horizon,
                        avail_features_full=avail_features_full,
                        params=params,
                    )

                    tabnet.fit(train, valid)
                    tabnet.fit(train_valid)

                    tabnet_pred.append(tabnet.predict(test))

                    logger.info(
                        f"Predict for {target}, horizon {horizon}, "
                        f"availability {avaliability} calculated, "
                        f"features: {features_type}"
                    )

    tabnet_pred_pl = pl.concat(tabnet_pred)
    os.makedirs("preds", exist_ok=True)
    tabnet_pred_pl.write_csv("preds/tabnet_pred_test.csv")


if __name__ == "__main__":
    run_main_tabnet()
