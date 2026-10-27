"""Put Mermaid forecast charts from the dashboard into README.md (GitHub draws them on the repo page).

    python -m pipelines.build_readme_charts
    python -m pipelines.build_readme_charts --dashboard dashboard/macro_dashboard.html --readme README.md
"""

from __future__ import annotations

import argparse

from forecasting.readme_charts import ReadmeCharts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dashboard", default="dashboard/macro_dashboard.html")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    path = ReadmeCharts.from_dashboard(args.dashboard).write_readme(args.readme)
    print(f"charts written to {path}")


if __name__ == "__main__":
    main()
