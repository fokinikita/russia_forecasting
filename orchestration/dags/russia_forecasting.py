"""Weekly real-time forecasting run: every Monday 05:00 Moscow time.

    resolve_run_date ─┬─ cbr_forecasts ──────────────────────────────────────────────┐
                      └─ collect_data ─ backtest_gb ─ backtest_ngb ─ backtest_tabnet  │
                           ─ backtest_dfm ─ backtest_mfbvar ─ evaluate_backtest      │
                           ─ forecast_future ─────────────────────────────── build_dashboard

Every task runs `python -m service.cli <stage> --date <run date>` in the project environment: the stage
pulls its inputs from S3 and pushes its results to S3 (folders <stage>/<run date>/, see service/layout.py),
so a failed task can simply be cleared and re-run. Tasks run one at a time (heavy CPU work on a shared
server). A failed backtest of one model does not stop the others: evaluation uses the models that finished.

Manual run ("Trigger DAG w/ config"): run_date (default = today), profile "quick" for a ~15 min test,
prefix "_smoke" to write into a separate folder of the bucket, models, force.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG, Param

TZ = pendulum.timezone("Europe/Moscow")
MODELS = ["gb", "ngb", "tabnet", "dfm", "mfbvar"]
PYTHON = "/opt/venv/bin/python"
PROJECT_DIR = "/app"
RUN_DATE = "{{ ti.xcom_pull(task_ids='resolve_run_date') }}"
OPTIONS = (
    "--profile {{ params.profile }}"
    "{% if params.prefix %} --prefix {{ params.prefix }}{% endif %}"
    "{% if params.force %} --force{% endif %}"
)


def resolve_run_date(params: dict, dag_run=None, **_) -> str:
    """Folder date of this run: the `run_date` param, else the day the run was triggered (Moscow)."""
    if params.get("run_date"):
        return datetime.strptime(params["run_date"], "%Y-%m-%d").strftime("%Y-%m-%d")
    moment = getattr(dag_run, "run_after", None) or getattr(dag_run, "logical_date", None) or pendulum.now(TZ)
    return pendulum.instance(moment).in_timezone(TZ).strftime("%Y-%m-%d")


def stage(task_id: str, command: str, timeout_hours: float, retries: int = 0, **kwargs) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"{PYTHON} -m service.cli {command} --date {RUN_DATE} {OPTIONS}",
        cwd=PROJECT_DIR,
        execution_timeout=timedelta(hours=timeout_hours),
        retries=retries,
        retry_delay=timedelta(minutes=10),
        **kwargs,
    )


with DAG(
    dag_id="russia_forecasting_weekly",
    description="IEP data -> backtest (retro metrics) -> 2-year forecasts -> CBR forecasts -> dashboard, all in S3",
    schedule="0 5 * * 1",
    start_date=pendulum.datetime(2026, 10, 1, tz=TZ),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    dagrun_timeout=timedelta(hours=20),
    default_args={"owner": "forecasting"},
    tags=["forecasting", "iep", "s3"],
    params={
        "run_date": Param("", type="string", description="YYYY-MM-DD, empty = today (Moscow)"),
        "profile": Param("full", enum=["full", "quick"], description="quick = small grid to test the service"),
        "prefix": Param("", type="string", description="root folder in the bucket, e.g. _smoke for tests"),
        "models": Param(MODELS, type="array", description="models to run"),
        "force": Param(False, type="boolean", description="recompute results that already exist in S3"),
    },
    doc_md=__doc__,
) as dag:
    run_date = PythonOperator(task_id="resolve_run_date", python_callable=resolve_run_date)

    cbr = stage("cbr_forecasts", "cbr", timeout_hours=1, retries=2)
    collect = stage("collect_data", "collect", timeout_hours=2, retries=2)

    backtests = []
    for model in MODELS:
        backtests.append(
            BashOperator(
                task_id=f"backtest_{model}",
                bash_command=(
                    f"{{% if '{model}' in params.models %}}"
                    f"{PYTHON} -m service.cli backtest --model {model} --date {RUN_DATE} {OPTIONS}"
                    f"{{% else %}}echo '{model} is not in params.models, skipped'{{% endif %}}"
                ),
                cwd=PROJECT_DIR,
                execution_timeout=timedelta(hours=8),
                retries=1,
                retry_delay=timedelta(minutes=5),
                trigger_rule="all_done",  # one failed model must not stop the others
            )
        )

    evaluate = stage("evaluate_backtest", "evaluate", timeout_hours=1, retries=1, trigger_rule="all_done")
    future = stage("forecast_future", "future --models {{ params.models | join(' ') }}", timeout_hours=6, retries=1)
    # CBR scraping is optional for the dashboard: it is built with the forecasts saved earlier
    dashboard = stage("build_dashboard", "dashboard", timeout_hours=1, retries=1, trigger_rule="all_done")

    run_date >> [cbr, collect]
    collect >> backtests[0]
    for previous, following in zip(backtests, backtests[1:]):
        previous >> following
    backtests[-1] >> evaluate >> future >> dashboard
    cbr >> dashboard
