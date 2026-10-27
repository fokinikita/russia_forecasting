"""Mermaid charts of the dashboard forecast for README.md.

GitHub does not run JavaScript in README, but it draws Mermaid diagrams, so the main forecast
(consensus, y/y growth with the 25-75% range of its members) is shown there as `xychart-beta` charts.
The numbers are taken from the built dashboard HTML, so README matches the published example page:

    python -m pipelines.build_readme_charts
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import attrs
import numpy as np

START_MARK = "<!-- readme-charts:start -->"
END_MARK = "<!-- readme-charts:end -->"
DATA_PREFIX = "const DATA = "  # how dashboard/template.html embeds the payload
# lower and upper bound of the range, then the fact/forecast line (drawn last, on top)
PALETTE = "#f59e0b, #f59e0b, #4f46e5"


def _quarter(d: date) -> str:
    return f"{d.year % 100:02d}Q{(d.month - 1) // 3 + 1}"


def _quarter_long(d: date) -> str:
    return f"{['I', 'II', 'III', 'IV'][(d.month - 1) // 3]} кв. {d.year}"


def _pct(log_growth: float) -> float:
    """Models forecast log growth; y/y in % as the dashboard shows it."""
    return (np.exp(log_growth) - 1) * 100


def _add_quarters(d: date, n: int) -> date:
    months = d.year * 12 + d.month - 1 + 3 * n
    return date(months // 12, months % 12 + 1, 1)


@attrs.define
class TargetChart:
    """Fact y/y growth for the last quarters and the 8-quarter forecast of one target."""

    name: str
    fact: list[tuple[date, float]]  # (quarter, y/y %)
    forecast: list[tuple[date, float, float, float]]  # (quarter, y/y %, low %, high %)

    def mermaid(self) -> str:
        labels = [_quarter(d) for d, _ in self.fact] + [f"{_quarter(d)}*" for d, *_ in self.forecast]
        main = [v for _, v in self.fact] + [v for _, v, _, _ in self.forecast]
        # Mermaid lines cannot have gaps: over the fact quarters the range lines repeat the fact
        # and hide under it, so the range "fans out" from the last fact
        fact = [v for _, v in self.fact]
        low = fact + [lo for *_, lo, _ in self.forecast]
        high = fact + [hi for *_, hi in self.forecast]
        values = main + low + high
        y_min, y_max = np.floor(min(values) - 1), np.ceil(max(values) + 1)
        fmt = lambda xs: "[" + ", ".join(f"{x:.1f}" for x in xs) + "]"  # noqa: E731
        return "\n".join(
            [
                "```mermaid",
                "%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '" + PALETTE + "'}}}}%%",
                "xychart-beta",
                f'    title "{self.name}, г/г, %"',
                "    x-axis [" + ", ".join(f'"{x}"' for x in labels) + "]",
                f'    y-axis "г/г, %" {y_min:.0f} --> {y_max:.0f}',
                f"    line {fmt(low)}",
                f"    line {fmt(high)}",
                f"    line {fmt(main)}",
                "```",
            ]
        )

    def table(self) -> str:
        rows = ["| квартал | прогноз г/г, % | диапазон 25–75%, % |", "|---|---:|---:|"]
        rows += [f"| {_quarter_long(d)} | {v:+.1f} | {lo:+.1f} … {hi:+.1f} |" for d, v, lo, hi in self.forecast]
        return "\n".join(rows)


@attrs.define
class ReadmeCharts:
    """Reads the dashboard payload and writes the chart block between markers in README."""

    payload: dict
    n_fact: int = 12  # fact quarters shown before the forecast

    @classmethod
    def from_dashboard(cls, html_path: Path | str, **kwargs) -> "ReadmeCharts":
        html = Path(html_path).read_text(encoding="utf-8")
        start = html.index(DATA_PREFIX) + len(DATA_PREFIX)
        payload, _ = json.JSONDecoder().raw_decode(html[start:])
        return cls(payload, **kwargs)

    @property
    def spec_key(self) -> str:
        return f"consensus/{self.payload['meta']['actual_availability']}"

    def target_chart(self, target: dict) -> TargetChart:
        history = [(date.fromisoformat(d), v) for d, v in target["history"]]
        levels = dict(history)
        yoy = [
            (d, (v / levels[_add_quarters(d, -4)] - 1) * 100)
            for d, v in history
            if _add_quarters(d, -4) in levels
        ]
        spec = target["specs"][self.spec_key]
        origin = date.fromisoformat(self.payload["meta"]["origin"])
        forecast = []
        for h, growth in enumerate(spec["future"]):
            if growth is None:
                continue
            members = spec.get("member_preds", [None] * len(spec["future"]))[h] or [growth]
            lo, hi = np.percentile(members, [25, 75])
            forecast.append((_add_quarters(origin, h), _pct(growth), _pct(lo), _pct(hi)))
        return TargetChart(name=target["name"], fact=yoy[-self.n_fact :], forecast=forecast)

    def markdown(self) -> str:
        meta = self.payload["meta"]
        created = date.fromisoformat(meta["created"][:10])
        parts = [
            f"Консенсус-прогноз (медиана {meta['top_k']} лучших спецификаций) из примера дашборда, "
            f"расчёт от {created:%d.%m.%Y}. Синяя линия — факт, далее прогноз (кварталы со `*`), "
            "оранжевые линии — диапазон 25–75% прогнозов спецификаций.",
        ]
        for i, target in enumerate(self.payload["targets"].values()):
            chart = self.target_chart(target)
            parts.append(
                f"<details{' open' if i == 0 else ''}>\n<summary><b>{chart.name}</b></summary>\n\n"
                f"{chart.mermaid()}\n\n{chart.table()}\n\n</details>"
            )
        return "\n\n".join(parts)

    def write_readme(self, readme_path: Path | str = "README.md") -> Path:
        readme_path = Path(readme_path)
        text = readme_path.read_text(encoding="utf-8")
        if START_MARK not in text or END_MARK not in text:
            raise ValueError(f"{readme_path} has no {START_MARK} ... {END_MARK} block")
        block = f"{START_MARK}\n{self.markdown()}\n{END_MARK}"
        pattern = re.escape(START_MARK) + ".*?" + re.escape(END_MARK)
        readme_path.write_text(re.sub(pattern, lambda _: block, text, flags=re.S), encoding="utf-8")
        return readme_path
