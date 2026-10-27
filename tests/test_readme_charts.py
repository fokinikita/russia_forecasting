from datetime import date

import numpy as np
import pytest

from forecasting.readme_charts import END_MARK, START_MARK, ReadmeCharts


def _payload() -> dict:
    quarters = [date(2024 + m // 12, m % 12 + 1, 1) for m in range(0, 24, 3)]  # 2024Q1 .. 2025Q4
    history = [[d.isoformat(), 100.0 + 10 * i] for i, d in enumerate(quarters)]
    growth = [np.log(1.02), np.log(1.03)]
    return {
        "meta": {"actual_availability": 1, "origin": "2026-01-01", "created": "2026-10-23T19:14", "top_k": 10},
        "targets": {
            "gdp": {
                "name": "ВВП",
                "history": history,
                "specs": {"consensus/1": {"future": growth, "member_preds": [[0.0, 0.01, 0.03], None]}},
            }
        },
    }


def test_chart_values():
    chart = ReadmeCharts(_payload(), n_fact=2).target_chart(_payload()["targets"]["gdp"])
    assert [d for d, _ in chart.fact] == [date(2025, 7, 1), date(2025, 10, 1)]
    assert chart.fact[-1][1] == pytest.approx((170 / 130 - 1) * 100)
    (d1, g1, lo1, hi1), (_, g2, lo2, hi2) = chart.forecast
    assert d1 == date(2026, 1, 1)
    assert (g1, g2) == pytest.approx((2.0, 3.0))
    assert lo1 == pytest.approx((np.exp(0.005) - 1) * 100) and hi1 == pytest.approx((np.exp(0.02) - 1) * 100)
    assert lo2 == hi2 == pytest.approx(3.0)  # no members: range collapses to the forecast
    assert "26Q1*" in chart.mermaid()


def test_write_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"head\n{START_MARK}\nold\n{END_MARK}\ntail\n", encoding="utf-8")
    ReadmeCharts(_payload()).write_readme(readme)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and "xychart-beta" in text
    assert text.startswith("head\n") and text.endswith("tail\n")
