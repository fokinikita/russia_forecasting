"""Bank of Russia medium-term forecasts (https://www.cbr.ru/dkp/cal_mp/, "Среднесрочный прогноз").

Every key-rate meeting with a forecast round publishes forecast_YYMMDD.pdf: one page with the table
"Основные параметры прогноза Банка России в рамках базового сценария" (growth, % y/y, ranges):

    2025
    2026 2027 2028 2029
    (факт)
    Валовой внутренний продукт 1,0 0,5–1,0 1,0–2,0 1,5–2,5 1,5–2,5
    Расходы на конечное потребление 2,9 3,0–3,5 ...
    – домашних хозяйств 3,6 3,0–3,5 ...
    Валовое накопление -4,9 (-4,0)–(-3,0) ...
    – основного капитала -0,4 (-1,0)–0,0 ...

The first year column is the fact / estimate. Rows of the four targets of the project are parsed:
gdp, cons (household consumption), inv (gross capital formation), inv_cap (gross fixed capital formation).
The service checks the page every week and parses only PDFs it has not seen yet.
"""

from __future__ import annotations

import io
import logging
import re
import time
from datetime import date
from typing import Optional

import attrs
import polars as pl

logger = logging.getLogger(__name__)

PAGE_URL = "https://www.cbr.ru/dkp/cal_mp/"
BASE_URL = "https://www.cbr.ru"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
LINK_RE = re.compile(r'href="(?P<href>[^"]*?/forecast_(?P<day>\d{6})\.pdf)"', re.IGNORECASE)

DASHES = "–—−‑"
NUMBER = r"\(?[-+]?\d+(?:[.,]\d+)?\)?"
VALUE_RE = re.compile(rf"^(?P<lo>{NUMBER})(?:-(?P<hi>{NUMBER}))?$")
YEAR_RE = re.compile(r"\b(20\d{2})\b")

SCHEMA = {
    "forecast_date": pl.Date,
    "target": pl.Utf8,
    "year": pl.Int64,
    "low": pl.Float64,
    "high": pl.Float64,
    "is_fact": pl.Boolean,
    "file": pl.Utf8,
    "url": pl.Utf8,
}


@attrs.frozen(order=True)
class CbrForecastRef:
    forecast_date: date
    url: str = attrs.field(order=False)

    @property
    def file(self) -> str:
        return f"forecast_{self.forecast_date:%y%m%d}.pdf"


# ----------------------------------------------------------------- scraping
@attrs.define
class CbrForecastScraper:
    page_url: str = PAGE_URL
    timeout: int = 60
    pause: float = 1.0  # seconds between PDF downloads

    def _get(self, url: str) -> bytes:
        import requests

        last_error = None
        for attempt in range(3):
            try:
                response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=self.timeout)
                response.raise_for_status()
                return response.content
            except Exception as error:  # cbr.ru sometimes drops connections
                last_error = error
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"cannot download {url}: {last_error}")

    @staticmethod
    def parse_links(html: str) -> list[CbrForecastRef]:
        refs = {}
        for match in LINK_RE.finditer(html):
            day = match["day"]
            forecast_date = date(2000 + int(day[:2]), int(day[2:4]), int(day[4:6]))
            href = match["href"]
            url = href if href.startswith("http") else BASE_URL + href
            refs.setdefault(forecast_date, CbrForecastRef(forecast_date, url))
        return sorted(refs.values(), reverse=True)

    def list_forecasts(self) -> list[CbrForecastRef]:
        refs = self.parse_links(self._get(self.page_url).decode("utf-8", errors="replace"))
        logger.info(f"cbr.ru: {len(refs)} medium-term forecasts, latest {refs[0].forecast_date if refs else None}")
        return refs

    def download(self, ref: CbrForecastRef) -> bytes:
        time.sleep(self.pause)
        return self._get(ref.url)


# ------------------------------------------------------------------ parsing
def _normalize(text: str) -> str:
    for dash in DASHES:
        text = text.replace(dash, "-")
    return text.replace(" ", " ")


def _number(token: str) -> float:
    return float(token.strip("()").replace(",", "."))


def parse_value(token: str) -> Optional[tuple[float, float]]:
    """'0,5-1,0' -> (0.5, 1.0); '(-4,0)-(-3,0)' -> (-4.0, -3.0); '-4,9' -> (-4.9, -4.9); '-' -> None."""
    match = VALUE_RE.match(token)
    if not match:
        return None
    lo = _number(match["lo"])
    hi = _number(match["hi"]) if match["hi"] else lo
    return (min(lo, hi), max(lo, hi))


# label (after dash normalization) -> target; the sub-rows "- домашних хозяйств" / "- основного капитала"
# are recognised by the main row right above them
MAIN_ROWS = {
    "Валовой внутренний продукт": "gdp",
    "Расходы на конечное потребление": "_consumption",
    "Валовое накопление": "inv",
}
SUB_ROWS = {("_consumption", "домашних хозяйств"): "cons", ("inv", "основного капитала"): "inv_cap"}


@attrs.define
class CbrForecastParser:
    def years(self, lines: list[str]) -> list[int]:
        """Years of the table header: between "(прирост в % ..." and the first data row."""
        start = next((i for i, line in enumerate(lines) if "прирост" in line.lower()), None)
        if start is None:
            return []
        years = []
        for line in lines[start + 1 :]:
            if re.match(r"^[А-ЯЁA-Z]", line) and not line.startswith("("):
                break  # first data row
            years += [int(y) for y in YEAR_RE.findall(line)]
        return years

    @staticmethod
    def _values(rest: str, n_years: int) -> Optional[list[Optional[tuple[float, float]]]]:
        tokens = rest.split()
        values = []
        for token in tokens:
            if token == "-":
                values.append(None)  # no data for this year
                continue
            value = parse_value(token)
            if value is None:
                return None  # not a row of values
            values.append(value)
        if not values or len(values) > n_years:
            return None
        return [None] * (n_years - len(values)) + values  # missing leading cells = no fact

    def parse_text(self, text: str, ref: CbrForecastRef) -> pl.DataFrame:
        lines = [line.strip() for line in _normalize(text).splitlines() if line.strip()]
        years = self.years(lines)
        rows, context = [], None
        if not years:
            return pl.DataFrame(schema=SCHEMA)
        for line in lines:
            target, rest = None, None
            for label, name in MAIN_ROWS.items():
                if line.startswith(label + " "):
                    rest = line[len(label) :]
                    if rest.strip().startswith("основного"):  # "Валовое накопление основного капитала ..."
                        name, rest = "inv_cap", rest.strip()[len("основного капитала") :]
                    target, context = name, name
                    break
            else:
                if line.startswith("-"):
                    body = line.lstrip("- ").strip()
                    for (parent, label), name in SUB_ROWS.items():
                        if context == parent and body.startswith(label):
                            target, rest = name, body[len(label) :]
                            break
                    if target is None and not body.startswith("в %"):
                        context = None
                elif not line.startswith("("):
                    context = None
            if target is None or target.startswith("_"):
                continue
            values = self._values(rest, len(years))
            if values is None:
                continue
            for i, (year, value) in enumerate(zip(years, values)):
                if value is None:
                    continue
                rows.append(
                    {
                        "forecast_date": ref.forecast_date,
                        "target": target,
                        "year": year,
                        "low": value[0],
                        "high": value[1],
                        "is_fact": i == 0,
                        "file": ref.file,
                        "url": ref.url,
                    }
                )
        frame = pl.DataFrame(rows, schema=SCHEMA).unique(["target", "year"], keep="first", maintain_order=True)
        missing = {"gdp", "cons", "inv", "inv_cap"} - set(frame["target"].to_list())
        if missing:
            logger.warning(f"{ref.file}: rows not found for {sorted(missing)}")
        return frame

    def parse_pdf(self, data: bytes, ref: CbrForecastRef) -> pl.DataFrame:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            # the table is on the first page; older reports have it on the second one
            for page in pdf.pages[:3]:
                frame = self.parse_text(page.extract_text() or "", ref)
                if frame.height:
                    return frame
        return pl.DataFrame(schema=SCHEMA)


def dashboard_payload(forecasts: pl.DataFrame, n: int = 3) -> list[dict]:
    """The `n` latest forecasts for the dashboard: [{"date", "url", "values": {target: {year: [lo, hi, is_fact]}}}]."""
    if forecasts.height == 0:
        return []
    dates = sorted(forecasts["forecast_date"].unique().to_list(), reverse=True)[:n]
    payload = []
    for day in dates:
        part = forecasts.filter(pl.col("forecast_date") == day)
        values: dict = {}
        for row in part.iter_rows(named=True):
            values.setdefault(row["target"], {})[str(row["year"])] = [row["low"], row["high"], row["is_fact"]]
        payload.append({"date": day.isoformat(), "url": part["url"][0], "file": part["file"][0], "values": values})
    return payload
