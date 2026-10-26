"""Service parts without network: Bank of Russia table parser, CSV <-> parquet storage format, model sink."""

from datetime import date

import polars as pl
import pytest

from forecasting.experiment import ExperimentConfig, ModelSink
from service.cbr import CbrForecastParser, CbrForecastRef, CbrForecastScraper, dashboard_payload, parse_value
from service.storage import csv_to_parquet, is_csv_table, parquet_to_csv

PDF_TEXT = """Среднесрочный прогноз Банка России
по итогам заседания Совета директоров по ключевой ставке 23 октября 2026 года
Основные параметры прогноза Банка России в рамках базового сценария
(прирост в % к предыдущему году, если не указано иное)
2025
2026 2027 2028 2029
(факт)
Инфляция, в %, декабрь к декабрю предыдущего года 5,6 7,1–7,6 4,0 4,0 4,0
Ключевая ставка, в среднем за год, в % годовых 19,2 14,61 12,5–13,5 9,0–10,0 7,5–8,5
Валовой внутренний продукт 1,0 0,5–1,0 1,0–2,0 1,5–2,5 1,5–2,5
– в %, IV квартал к IV кварталу предыдущего года 1,0 0,0–1,5 1,0–2,0 1,5–2,5 1,5–2,5
Расходы на конечное потребление 2,9 3,0–3,5 0,5–1,5 1,0–2,0 1,5–2,5
– домашних хозяйств 3,6 3,0–3,5 0,5–1,5 1,0–2,0 1,5–2,5
Валовое накопление -4,9 (-4,0)–(-3,0) 1,5–3,5 1,5–3,5 1,0–3,0
– основного капитала -0,4 (-1,0)–0,0 1,0–3,0 1,5–3,5 1,0–3,0
Экспорт – 2 (-1,0)–1,0 0,5–2,5 1,5–3,5 1,0–3,0
"""


@pytest.mark.parametrize(
    "token, expected",
    [("0,5-1,0", (0.5, 1.0)), ("(-4,0)-(-3,0)", (-4.0, -3.0)), ("-4,9", (-4.9, -4.9)), ("(-1,0)-(+1,0)", (-1.0, 1.0)), ("abc", None)],
)
def test_parse_value(token, expected):
    assert parse_value(token) == expected


def test_parse_table():
    ref = CbrForecastRef(date(2026, 10, 23), "https://www.cbr.ru/x/forecast_261023.pdf")
    frame = CbrForecastParser().parse_text(PDF_TEXT, ref)
    values = {(r["target"], r["year"]): (r["low"], r["high"], r["is_fact"]) for r in frame.iter_rows(named=True)}
    assert values[("gdp", 2025)] == (1.0, 1.0, True)
    assert values[("gdp", 2026)] == (0.5, 1.0, False)
    assert values[("cons", 2027)] == (0.5, 1.5, False)  # households, not total final consumption
    assert values[("inv", 2026)] == (-4.0, -3.0, False)
    assert values[("inv_cap", 2026)] == (-1.0, 0.0, False)
    assert set(frame["target"]) == {"gdp", "cons", "inv", "inv_cap"}
    assert frame.height == 20

    payload = dashboard_payload(frame)
    assert payload[0]["date"] == "2026-10-23"
    assert payload[0]["values"]["gdp"]["2027"] == [1.0, 2.0, False]


def test_parse_links():
    html = (
        '<a href="/Content/Document/File/198897/forecast_261023.pdf">Среднесрочный прогноз</a>'
        '<a href="/Content/Document/File/187047/comment_26022026.pdf">Комментарий</a>'
        '<a href="/Collection/Collection/File/48891/forecast_240216.pdf">x</a>'
    )
    refs = CbrForecastScraper.parse_links(html)
    assert [r.forecast_date for r in refs] == [date(2026, 10, 23), date(2024, 2, 16)]
    assert refs[0].url == "https://www.cbr.ru/Content/Document/File/198897/forecast_261023.pdf"
    assert refs[0].file == "forecast_261023.pdf"


def test_csv_parquet_round_trip(tmp_path):
    frame = pl.DataFrame({"run_id": ["2026-10-22"], "target_date": ["2026-07-01"], "rmse": [0.0123456789], "n": [12]})
    frame.write_csv(tmp_path / "metrics.csv")
    csv_to_parquet(tmp_path / "metrics.csv", tmp_path / "metrics.parquet")
    assert pl.read_parquet(tmp_path / "metrics.parquet")["run_id"].dtype == pl.Utf8  # run ids stay text
    parquet_to_csv(tmp_path / "metrics.parquet", tmp_path / "back.csv")
    assert (tmp_path / "back.csv").read_text() == (tmp_path / "metrics.csv").read_text()
    assert is_csv_table("failed_specs_gb") and is_csv_table("forecasts") and not is_csv_table("forecasts_gb")


def test_model_sink_key(tmp_path):
    cfg = ExperimentConfig("gb", 2010, "short", "classic", 1, target="gdp", horizon=3)
    sink = ModelSink(tmp_path, frozenset({"gb/2010/short/classic/a1/gdp"}))
    assert sink.wants(cfg, "gdp") and not sink.wants(cfg, "cons")
    sink.save_direct(cfg, {"any": "object"}, ("x_m1",), date(2026, 4, 1))
    assert (tmp_path / "gb" / "gb__2010__short__classic__a1__gdp__h3.joblib").exists()
