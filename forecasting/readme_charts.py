"""Clickable dashboard for README.md, built from the dashboard HTML.

GitHub does not run JavaScript on the repo page, so the HTML dashboard cannot work there. It does
support collapsible `<details>` blocks and draws Mermaid `xychart-beta` charts, so README gets the
same menu as the dashboard: indicator -> model -> tab (forecast y/y, levels, backtest).
The numbers are taken from the built dashboard, so README matches the published example page:

    python -m pipelines.build_readme_charts
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

import attrs
import numpy as np

START_MARK = "<!-- readme-charts:start -->"
END_MARK = "<!-- readme-charts:end -->"
DATA_PREFIX = "const DATA = "  # how dashboard/template.html embeds the payload

# same names and colours as dashboard/template.html
MODELS = {
    "consensus": "Консенсус (медиана топ-10)",
    "best": "Лучшая спецификация (автовыбор)",
    "gb": "CatBoost",
    "ngb": "NGBoost",
    "dfm": "DFM",
    "tabnet": "TabNet (нейросеть)",
    "mfbvar": "MF-BVAR",
}
SELECTED_MODELS = ("consensus", "best")
VARSET_NAMES = {"short": "короткий", "medium": "средний", "all": "полный"}
FTYPE_NAMES = {"classic": "классические", "rolling": "скользящие"}
FACT_COLOR, FORECAST_COLOR, H4_COLOR = "#4f46e5", "#f59e0b", "#db2777"
BAND_Q = (25, 75)
RETRO_HORIZONS = (1, 4)


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


def _spec_text(key: str) -> str:
    model, start, varset, ftype, _ = key.split("/")
    return f"{MODELS[model]} · с {start} г. · {VARSET_NAMES[varset]} набор · {FTYPE_NAMES[ftype]} признаки"


@attrs.define
class Line:
    values: list[float]
    color: str


@attrs.define
class MermaidChart:
    """One `xychart-beta` line chart. Lines are drawn in order, the last one on top."""

    title: str
    y_label: str
    labels: list[str]
    lines: list[Line]
    digits: int = 1

    def render(self) -> str:
        values = [v for line in self.lines for v in line.values]
        pad = (max(values) - min(values)) * 0.1 or 1
        y_min, y_max = min(values) - pad, max(values) + pad
        fmt = lambda xs: "[" + ", ".join(f"{x:.{self.digits}f}" for x in xs) + "]"  # noqa: E731
        palette = ", ".join(line.color for line in self.lines)
        return "\n".join(
            [
                "```mermaid",
                "%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '" + palette + "'}}}}%%",
                "xychart-beta",
                f'    title "{self.title}"',
                "    x-axis [" + ", ".join(f'"{x}"' for x in self.labels) + "]",
                f'    y-axis "{self.y_label}" {y_min:.{self.digits}f} --> {y_max:.{self.digits}f}',
                *[f"    line {fmt(line.values)}" for line in self.lines],
                "```",
            ]
        )


@attrs.define
class Forecast:
    """Forecast of one model for one target: future path, backtest and metrics."""

    target_name: str
    model: str
    description: str
    history: list[tuple[date, float]]  # levels, bn rub
    test_dates: list[date]
    test_fact: list[Optional[float]]  # y/y %
    future: list[tuple[date, float, float, float, float]]  # (quarter, y/y %, low %, high %, level)
    test: dict[int, list[Optional[float]]]  # horizon -> backtest y/y % per test date
    metrics: list[Optional[dict]]  # per horizon
    naive_mae: dict[int, float]
    n_fact: int = 8

    @property
    def fact_yoy(self) -> list[tuple[date, float]]:
        levels = dict(self.history)
        rows = [(d, (v / levels[_add_quarters(d, -4)] - 1) * 100) for d, v in self.history if _add_quarters(d, -4) in levels]
        return rows[-self.n_fact :]

    def _labels(self, fact_dates: list[date]) -> list[str]:
        return [_quarter(d) for d in fact_dates] + [f"{_quarter(d)}*" for d, *_ in self.future]

    def _fan(self, fact: list[float], main: list[float], low: list[float], high: list[float]) -> list[Line]:
        # Mermaid lines cannot have gaps: over the fact quarters the range lines repeat the fact and
        # hide under it, so the range "fans out" from the last fact
        lines = [Line(fact + main, FACT_COLOR)]
        if any(lo != hi for lo, hi in zip(low, high)):
            lines = [Line(fact + low, FORECAST_COLOR), Line(fact + high, FORECAST_COLOR)] + lines
        return lines

    def yoy_chart(self) -> str:
        fact = self.fact_yoy
        chart = MermaidChart(
            title=f"{self.target_name}, г/г, %",
            y_label="г/г, %",
            labels=self._labels([d for d, _ in fact]),
            lines=self._fan(
                [v for _, v in fact],
                [f[1] for f in self.future],
                [f[2] for f in self.future],
                [f[3] for f in self.future],
            ),
        )
        rows = ["| квартал | прогноз г/г, % | диапазон 25–75%, % | уровень, трлн ₽ |", "|---|---:|---:|---:|"]
        rows += [
            f"| {_quarter_long(d)} | {v:+.1f} | " + (f"{lo:+.1f} … {hi:+.1f}" if lo != hi else "—") + f" | {lvl / 1000:.2f} |"
            for d, v, lo, hi, lvl in self.future
        ]
        return chart.render() + "\n\n" + "\n".join(rows)

    def level_chart(self) -> str:
        fact = self.history[-self.n_fact :]
        levels = dict(self.history)
        low, high = [], []
        for d, _, lo, hi, _ in self.future:
            base = levels[_add_quarters(d, -4)]
            low.append(base * (1 + lo / 100) / 1000)
            high.append(base * (1 + hi / 100) / 1000)
            levels[d] = next(f[4] for f in self.future if f[0] == d)
        chart = MermaidChart(
            title=f"{self.target_name}, трлн ₽ (цены 2021 г.)",
            y_label="трлн ₽",
            labels=self._labels([d for d, _ in fact]),
            lines=self._fan([v / 1000 for _, v in fact], [f[4] / 1000 for f in self.future], low, high),
            digits=2,
        )
        return chart.render()

    def backtest_chart(self) -> str:
        # h = 4 under h = 1 under the fact; a horizon with gaps is left out (Mermaid lines cannot have gaps)
        lines = [Line(self.test.get(h, [None]), color) for h, color in ((4, H4_COLOR), (1, FORECAST_COLOR))]
        lines = [line for line in lines + [Line(self.test_fact, FACT_COLOR)] if None not in line.values]
        chart = MermaidChart(
            title=f"{self.target_name}: ретроспектива, г/г, %",
            y_label="г/г, %",
            labels=[_quarter(d) for d in self.test_dates],
            lines=lines,
        )
        rows = ["| горизонт | MAE модели, п.п. | MAE наивного, п.п. | модель / наивный |", "|---|---:|---:|---:|"]
        for h, m in enumerate(self.metrics, start=1):
            if m is None:
                continue
            naive = self.naive_mae[h]
            status = "🟢" if m["mae"] < naive else "🔴"
            rows.append(f"| h = {h} | {m['mae']:.2f} | {naive:.2f} | {m['mae'] / naive:.0%} {status} |")
        legend = (
            "Синяя — факт, оранжевая — прогноз на 1 квартал вперёд (h = 1), розовая — на 4 квартала (h = 4); "
            "прогнозы строились только по данным, известным на момент прогноза."
        )
        return f"{legend}\n\n{chart.render()}\n\n" + "\n".join(rows)


@attrs.define
class ReadmeCharts:
    """Reads the dashboard payload and writes the clickable block between markers in README."""

    payload: dict
    n_fact: int = 8  # fact quarters before the forecast (more do not fit as x-axis labels)

    @classmethod
    def from_dashboard(cls, html_path: Path | str, **kwargs) -> "ReadmeCharts":
        html = Path(html_path).read_text(encoding="utf-8")
        start = html.index(DATA_PREFIX) + len(DATA_PREFIX)
        payload, _ = json.JSONDecoder().raw_decode(html[start:])
        return cls(payload, **kwargs)

    @property
    def availability(self) -> int:
        return self.payload["meta"]["actual_availability"]

    def spec_key(self, target: dict, model: str) -> Optional[str]:
        """Consensus / best by their keys; a single model by its spec with the lowest mean MAE."""
        if model in SELECTED_MODELS:
            key = f"{model}/{self.availability}"
            return key if key in target["specs"] else None
        candidates = []
        for key, spec in target["specs"].items():
            if key.startswith(f"{model}/") and key.endswith(f"/{self.availability}"):
                maes = [m["mae"] for m in spec["metrics"] if m]
                if maes and None not in spec["future"]:
                    candidates.append((np.mean(maes), key))
        return min(candidates)[1] if candidates else None

    def describe(self, target: dict, model: str, key: str) -> str:
        spec = target["specs"][key]
        if model == "consensus":
            return f"медиана {self.payload['meta']['top_k']} лучших на ретроспективе спецификаций на каждом горизонте"
        if model == "best":
            chosen = spec.get("chosen") or next((m for m in spec.get("members") or [] if m), None)
            return f"автовыбор: {_spec_text(chosen)}" if chosen else "лучшая на ретроспективе спецификация"
        return f"лучшая на ретроспективе спецификация модели: {_spec_text(key)}"

    def forecast(self, target: dict, model: str) -> Optional[Forecast]:
        key = self.spec_key(target, model)
        if key is None:
            return None
        spec = target["specs"][key]
        history = [(date.fromisoformat(d), v) for d, v in target["history"]]
        levels = dict(history)
        origin = date.fromisoformat(self.payload["meta"]["origin"])
        future = []
        for h, growth in enumerate(spec["future"]):
            if growth is None:
                continue
            d = _add_quarters(origin, h)
            members = (spec.get("member_preds") or [None] * len(spec["future"]))[h] or [growth]
            lo, hi = np.percentile(members, BAND_Q)
            levels[d] = levels[_add_quarters(d, -4)] * np.exp(growth)
            future.append((d, _pct(growth), _pct(lo), _pct(hi), levels[d]))
        to_pct = lambda xs: [None if x is None else _pct(x) for x in xs]  # noqa: E731
        return Forecast(
            target_name=target["name"],
            model=model,
            description=self.describe(target, model, key),
            history=history,
            test_dates=[date.fromisoformat(d) for d in self.payload["test_dates"]],
            test_fact=to_pct(target["test_fact"]),
            future=future,
            test={h: to_pct(spec["test"][h - 1]) for h in RETRO_HORIZONS if h <= len(spec["test"])},
            metrics=spec["metrics"],
            naive_mae={int(h): v["mae"] for h, v in target["naive"].items()},
            n_fact=self.n_fact,
        )

    @staticmethod
    def _details(summary: str, body: str, is_open: bool = False) -> str:
        return f"<details{' open' if is_open else ''}>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"

    def model_block(self, fc: Forecast, is_open: bool) -> str:
        tabs = "\n\n".join(
            [
                self._details("📈 <b>Прогноз г/г</b>", fc.yoy_chart(), is_open=True),
                self._details("💰 <b>Уровни</b>", fc.level_chart()),
                self._details("🔁 <b>Ретроспектива</b> (качество на истории)", fc.backtest_chart()),
            ]
        )
        body = f"<sub>{fc.description}</sub>\n\n{tabs}"
        return self._details(f"<b>{MODELS[fc.model]}</b>", _indent(body), is_open)

    def markdown(self) -> str:
        parts = [
            "Нажимайте: показатель → модель → вкладка. Кварталы со `*` — прогноз, "
            "оранжевые линии — диапазон 25–75% прогнозов спецификаций, графики можно приближать кнопками справа."
        ]
        for i, target in enumerate(self.payload["targets"].values()):
            models = [fc for model in MODELS if (fc := self.forecast(target, model)) is not None]
            body = "\n\n".join(self.model_block(fc, is_open=(j == 0)) for j, fc in enumerate(models))
            parts.append(self._details(f"📊 <b>{target['name']}</b>", _indent(body), is_open=(i == 0)))
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


def _indent(body: str) -> str:
    """Nested menus are shifted right with a blockquote, so the levels are visible."""
    return "\n".join(f"> {line}" if line else ">" for line in body.split("\n"))
