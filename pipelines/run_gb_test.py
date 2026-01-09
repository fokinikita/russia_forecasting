import os
import polars as pl
import logging

from preprocess_data.datae2e import DataE2E
from models.gb import GB

import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_main_gb() -> None:
    horizon_grid = range(1, config.HORIZON + 1)
    features_type_grid = ['rolling', 'd12']
    avaliability_grid = range(1, config.MAX_AVALIABILITY + 1)
    target_grid = ['gdp_log_d4', 'cons_log_d4', 'inv_log_d4', 'inv_cap_log_d4']

    train, valid, train_valid, test, avail_features_full = DataE2E().run()

    gb_pred = []
    for features_type in features_type_grid:
        for target in target_grid:
            for horizon in horizon_grid:
                for avaliability in avaliability_grid:
                    params = {
                        "iterations": config.START_ITERATIONS,
                    }

                    gb = GB(
                        features_type=features_type,
                        avaliability=avaliability,
                        target_name=target,
                        horizon=horizon,
                        avail_features_full=avail_features_full,
                        params=params,
                    )

                    gb.fit(train, valid, early_stopping=config.EARLY_STOPPING_ROUNDS)

                    gb.fit(train_valid)

                    gb_pred.append(gb.predict(test))
                    logger.info(f"Predict for {target},"
                                f" horizon {horizon} with avaliability {avaliability} was calculated,"
                                f" features: {features_type}")

    gb_pred_pl = pl.concat(gb_pred)
    os.makedirs("preds", exist_ok=True)
    gb_pred_pl.write_csv('preds/gb_pred_test.csv')

if __name__ == "__main__":
    run_main_gb()
