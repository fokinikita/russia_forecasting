import os

import polars as pl

from metrics.metrics import MetricsCalculator
from preprocess_data.datae2e import DataE2E


def run_metrics(calculate_mfbvar: bool = True, calculate_dfm: bool = True) -> None:
    logger.info("Start calculating metrics")

    if calculate_mfbvar:
        logger.info("MFBVAR metrics calculation will be performed too")
    if calculate_dfm:
        logger.info("DFM metrics calculation will be performed too")

    train, valid, train_valid, test, avail_features_full = DataE2E().run()

    gb_pred_pl = pl.read_csv("preds/gb_pred_test.csv").with_columns(
        pl.col("date").cast(pl.Date)
    )

    ngb_pred_pl = pl.read_csv("preds/ngb_pred_test.csv").with_columns(
        pl.col("date").cast(pl.Date)
    )

    tabnet_pred_pl = pl.read_csv("preds/tabnet_pred_test.csv").with_columns(
        pl.col("date").cast(pl.Date)
    )

    preds_list = [gb_pred_pl, ngb_pred_pl, tabnet_pred_pl]
    ml_model_names = ["gb", "ngb", "tabnet"]
    target_names = ["gdp_log_d4", "cons_log_d4", "inv_log_d4", "inv_cap_log_d4"]

    metrics = MetricsCalculator(
        preds_list, ml_model_names, target_names, pl.concat([train, valid, test])
    ).get_metrics(calculate_mfbvar, calculate_dfm)

    os.makedirs("preds", exist_ok=True)
    metrics.write_csv("preds/metrics.csv")

    logger.info("All metrics was calculated")


if __name__ == "__main__":
    run_metrics()
