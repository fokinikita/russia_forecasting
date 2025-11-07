import polars as pl
from preprocess_data.datae2e import DataE2E
from metrics.metrics import MetricsCalculator

def run_metrics() -> None:
    train, valid, train_valid, test, avail_features_full = DataE2E().run()

    gb_pred_pl = pl.read_csv('gb_pred_test.csv').with_columns(
        pl.col('date').cast(pl.Date)
    )

    ngb_pred_pl = pl.read_csv('ngb_pred_test.csv').with_columns(
        pl.col('date').cast(pl.Date)
    )

    tabnet_pred_pl = pl.read_csv('tabnet_pred_test.csv').with_columns(
        pl.col('date').cast(pl.Date)
    )

    preds_list = [gb_pred_pl, ngb_pred_pl, tabnet_pred_pl]
    model_names = ['gb', 'ngb', 'tabnet']
    target_names = ['gdp_log_d4', 'cons_log_d4', 'inv_log_d4', 'inv_cap_log_d4']

    metrics = MetricsCalculator(preds_list, model_names, target_names, pl.concat([train, valid, test])).get_metrics()
    matrics.write_csv('metrics.csv')

if __name__ == '__main__':
    run_metrics()