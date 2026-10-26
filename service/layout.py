"""Where the service keeps things: S3 folders per run date and the local working copies.

S3 (bucket root, or <S3_PREFIX>/ for test runs), <date> = date the run was triggered (YYYY-MM-DD, Moscow):

    raw_data/<date>/            IEP data: quarterly / monthly / daily (+ raw_*) .parquet, metadata.parquet
    retro_metrics/<date>/       backtest: forecasts_<model>.parquet, errors, metrics, ranking, best,
                                selected_specs, rolling, failed_specs .parquet, run_info.json
    forecasts/<date>/           future: forecasts_<model>.parquet, forecasts, best, consensus, selected .parquet
    models/<date>/<model>/      fitted models of the auto-chosen specs (joblib; DFM -> parameters parquet)
    cbr_forecasts/pdf/          Bank of Russia medium-term forecast PDFs (downloaded once)
    cbr_forecasts/all_forecasts.parquet   every parsed CBR forecast
    cbr_forecasts/<date>/cbr_forecasts.parquet   what was known on the run date
    dashboard/<date>/macro_dashboard.html, dashboard/latest/macro_dashboard.html
    history/metrics_history.parquet, history/forecasts_history.parquet   all runs
    airflow_logs/               Airflow task logs
"""

from __future__ import annotations

from pathlib import Path

import attrs

import config

RAW, RETRO, FORECASTS, MODELS, CBR, DASHBOARD, HISTORY = (
    "raw_data",
    "retro_metrics",
    "forecasts",
    "models",
    "cbr_forecasts",
    "dashboard",
    "history",
)
DASHBOARD_FILE = "macro_dashboard.html"


@attrs.frozen
class RunLayout:
    run_date: str
    data_dir: Path = attrs.field(default=Path(config.IEP_DATA_DIR), converter=Path)
    results_dir: Path = attrs.field(default=Path(config.RT_RESULTS_DIR), converter=Path)

    # ------------------------------------------------------------------ S3
    def s3(self, stage: str, run_date: str | None = None) -> str:
        return f"{stage}/{run_date or self.run_date}"

    @staticmethod
    def s3_history() -> str:
        return HISTORY

    # --------------------------------------------------------------- local
    def run_dir(self, run_date: str | None = None) -> Path:
        return self.results_dir / "runs" / (run_date or self.run_date)

    def backtest_dir(self, run_date: str | None = None) -> Path:
        return self.run_dir(run_date) / "backtest"

    def future_dir(self, run_date: str | None = None) -> Path:
        return self.run_dir(run_date) / "future"

    def models_dir(self) -> Path:
        return self.run_dir() / "models"

    def dashboard_path(self) -> Path:
        return self.run_dir() / "dashboard" / DASHBOARD_FILE

    def cbr_dir(self) -> Path:
        return self.results_dir / "cbr"
