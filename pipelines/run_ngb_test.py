import os
import polars as pl
import logging

from preprocess_data.datae2e import DataE2E
from models.ngb import NGB

import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_main_ngb() -> None:
    horizon_grid = range(1, config.HORIZON + 1)
    features_type_grid = ['rolling', 'd12']
    avaliability_grid = range(1, config.MAX_AVALIABILITY + 1)
    target_grid = ['gdp_log_d4', 'cons_log_d4', 'inv_log_d4', 'inv_cap_log_d4']

    train, valid, train_valid, test, avail_features_full = DataE2E().run()

    ngb_pred = []
    for features_type in features_type_grid:
        for target in target_grid:
            for horizon in horizon_grid:
                for avaliability in avaliability_grid:
                    params = {
                        "n_estimators": config.START_ITERATIONS_NGB,
                        "learning_rate": config.LEARNING_RATE_NGB,
                        "verbose": False,
                    }

                    ngb = NGB(
                        features_type=features_type,
                        avaliability=avaliability,
                        target_name=target,
                        horizon=horizon,
                        avail_features_full=avail_features_full,
                        params=params,
                    )

                    ngb.fit(train, valid, early_stopping=config.EARLY_STOPPING_ROUNDS_NGB)
                    ngb.fit(train_valid)

                    ngb_pred.append(ngb.predict(test))
                    logger.info(f"Predict for {target},"
                                f" horizon {horizon} with avaliability {avaliability} was calculated,"
                                f" features: {features_type}")

    ngb_pred_pl = pl.concat(ngb_pred)
    os.makedirs("preds", exist_ok=True)
    ngb_pred_pl.write_csv('preds/ngb_pred_test.csv')

if __name__ == "__main__":
    run_main_ngb()
