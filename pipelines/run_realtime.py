"""Real-time forecasting on IEP data: backtest metrics + 2-year forecasts.

    python -m pipelines.run_realtime                       # everything (refresh, backtest, future)
    python -m pipelines.run_realtime --no-refresh          # use the saved IEP data
    python -m pipelines.run_realtime --models gb dfm       # only some models
    python -m pipelines.run_realtime --only future         # only future forecasts
    python -m pipelines.run_realtime --run-id 2026-10-22   # continue / redo a run (checkpoints are reused)
    python -m pipelines.run_realtime --run-id 2026-10-22 --retry-failed --n-jobs 2   # re-run failed specs
    python -m pipelines.run_realtime --refit-every 1       # re-estimate every quarter (strict real time, ~4x slower)
"""

from __future__ import annotations

import argparse
import logging

import config
from forecasting.dashboard import build_dashboard
from forecasting.experiment import ExperimentSettings
from forecasting.pipeline import RealTimePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(config.RT_MODELS), choices=["gb", "ngb", "tabnet", "dfm", "mfbvar"])
    parser.add_argument("--only", choices=["backtest", "future"], help="run only one stage")
    parser.add_argument("--no-refresh", action="store_true", help="do not download IEP data")
    parser.add_argument("--run-id", help="results folder name (default: today)")
    parser.add_argument("--n-jobs", type=int, default=config.RT_N_JOBS)
    parser.add_argument("--no-resume", action="store_true", help="ignore checkpoints of the same run")
    parser.add_argument(
        "--refit-every",
        type=int,
        default=config.RT_REFIT_EVERY,
        help="re-estimate models every N origins (1 = every quarter)",
    )
    parser.add_argument(
        "--retry-failed", action="store_true", help="only re-run specifications that failed in this run"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("forecasting.dataset").setLevel(logging.WARNING)
    args = parse_args()

    kwargs = {
        "models": tuple(args.models),
        "n_jobs": args.n_jobs,
        "settings": ExperimentSettings(refit_every=args.refit_every),
    }
    if args.run_id:
        kwargs["run_id"] = args.run_id
    pipeline = RealTimePipeline(**kwargs)
    if args.retry_failed:
        pipeline.retry_failed()
        return
    pipeline.run(
        refresh=not args.no_refresh,
        backtest=args.only in (None, "backtest"),
        future=args.only in (None, "future"),
        resume=not args.no_resume,
    )
    if args.only in (None, "future"):
        try:
            build_dashboard(run_id=pipeline.run_id)
        except Exception as exc:  # the dashboard must never break the forecasting run
            logging.warning(f"dashboard was not built: {exc}")


if __name__ == "__main__":
    main()
