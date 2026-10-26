"""Command line of the service (Airflow runs exactly these commands, they also work by hand).

    python -m service.cli check                                  # S3 connection
    python -m service.cli collect   --date 2026-11-02
    python -m service.cli backtest  --date 2026-11-02 --model gb
    python -m service.cli evaluate  --date 2026-11-02
    python -m service.cli future    --date 2026-11-02
    python -m service.cli cbr       --date 2026-11-02
    python -m service.cli dashboard --date 2026-11-02
    python -m service.cli seed      --date 2026-10-22 --source-results /seed/results --source-data /seed/data

Options: --profile quick (small grid for a test run), --prefix _smoke (write under another root folder
of the bucket), --force (recompute what is already in S3), --n-jobs N.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import config
from service.settings import PROFILES
from service.stages import (
    BacktestStage,
    CbrStage,
    CollectStage,
    DashboardStage,
    EvaluateStage,
    FutureStage,
    SeedStage,
    StageContext,
)
from service.storage import S3Storage

STAGES = ("check", "collect", "backtest", "evaluate", "future", "cbr", "dashboard", "seed")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--date", default=datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d"))
    parser.add_argument("--model", choices=config.RT_MODELS, help="backtest: the model to run")
    parser.add_argument("--models", nargs="+", default=list(config.RT_MODELS), choices=config.RT_MODELS)
    parser.add_argument("--profile", default="full", choices=PROFILES)
    parser.add_argument("--prefix", default=None, help="root folder inside the bucket (default: S3_PREFIX env)")
    parser.add_argument("--n-jobs", type=int, default=config.RT_N_JOBS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-results", default="results")
    parser.add_argument("--source-data", default=config.IEP_DATA_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("forecasting.dataset").setLevel(logging.WARNING)
    for noisy in ("botocore", "boto3", "s3transfer", "urllib3", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args(argv)
    storage = S3Storage.from_env(args.prefix)

    if args.stage == "check":
        storage.put_bytes(b"ok", storage.key("_healthcheck.txt"), "text/plain")
        print(f"S3 ok: {storage.uri(storage.key(''))}, folders: {storage.folders(storage.key(''))}")
        return 0

    ctx = StageContext(args.date, storage, profile=args.profile, n_jobs=args.n_jobs, force=args.force)
    logging.info(f"stage {args.stage}, run date {args.date}, profile {args.profile}, s3 {storage.uri(storage.key(''))}")
    if args.stage == "collect":
        result = CollectStage(ctx).run()
    elif args.stage == "backtest":
        if not args.model:
            raise SystemExit("backtest needs --model")
        result = BacktestStage(ctx, args.model).run()
    elif args.stage == "evaluate":
        result = EvaluateStage(ctx).run()
    elif args.stage == "future":
        result = FutureStage(ctx, tuple(args.models)).run()
    elif args.stage == "cbr":
        result = CbrStage(ctx).run()
    elif args.stage == "dashboard":
        result = DashboardStage(ctx).run()
    else:
        result = SeedStage(ctx, args.source_results, args.source_data).run()
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
