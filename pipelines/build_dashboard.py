"""Build the HTML dashboard from the results of a real-time run.

    python -m pipelines.build_dashboard                          # latest finished run, availability from today
    python -m pipelines.build_dashboard --run-id 2026-10-22
    python -m pipelines.build_dashboard --as-of 2026-11-15       # pretend today is 15 Nov (availability 2)
    python -m pipelines.build_dashboard --out somewhere/report.html
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from forecasting.dashboard import build_dashboard


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", help="run folder in results/runs (default: latest finished)")
    parser.add_argument("--as-of", type=date.fromisoformat, help="date that sets data availability (default: today)")
    parser.add_argument("--out", default="dashboard/macro_dashboard.html")
    args = parser.parse_args()
    build_dashboard(run_id=args.run_id, as_of=args.as_of, out_path=args.out)


if __name__ == "__main__":
    main()
