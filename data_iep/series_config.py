"""Registry of every series collected from data.iep.ru.

Each SeriesSpec says *where* a variable lives (dataset uuid + dimension values that
identify one line inside the dataset) and *how* it must be turned into a "basic"
(level) series later on (`kind`).

The vintage id is never stored here: the latest vintage is resolved on every run.

kinds:
    level      - already a level (rubles, tonnes, %, index with fixed base, ...)
    chain_mom  - % to previous period (chained index)         -> I_t = I_{t-1} * x_t / 100
    chain_yoy  - % to same period of previous year            -> I_t = I_{t-12} * x_t / 100
    splice     - one piece of a series published in several parts (GDP in 2003/2008/2021
                 prices, unemployment 15-72 / 15+). Pieces sharing `target` are listed
                 newest first; older pieces extend the history backwards through
                 their y/y growth, so everything ends up in 2021 prices.
"""

from __future__ import annotations

from typing import Literal

import attrs

Frequency = Literal["Q", "M", "D"]
Kind = Literal["level", "chain_mom", "chain_yoy", "splice"]


@attrs.frozen
class SeriesSpec:
    name: str
    frequency: Frequency
    dataset_uuid: str
    dims: tuple[str, ...] = ()  # values that must all appear in the line selector
    unit_id: int | None = None
    kind: Kind = "level"
    search_query: str | None = None  # fallback to find a dataset the family endpoint misses
    target: str | None = None  # final column name for splice pieces


# ----------------------------------------------------------------- datasets
GDP = "911f4b10-def4-45b6-8dba-2ad649cf8e43"
IPI = "c1c92863-1827-405e-b3e4-dea782f57316"
IPI_SECTORS = "a2abf3ae-0dc3-4571-a315-f4cedce9ef5b"
AGRICULTURE = "f59719eb-5204-49a3-813f-c2bbd7c02697"
LIVESTOCK = "8a0846b8-dbd7-4197-86e8-1ab65f7ee342"
FREIGHT = "a5ccca13-0170-45b4-9c93-d19433f0cb3d"
RAIL = "b8644b57-b5eb-4491-b83a-ce3799bb3436"
CONSTRUCTION = "25a96904-cf91-414d-ac9c-5c724f2fd42c"
HOUSING = "50ef8b82-b651-407d-a4aa-08a277a3fa2e"
RETAIL = "536514f9-aa12-4acb-870d-ec4278d00721"
SERVICES = "0e663ba8-0d9e-4b8c-a405-71450c2c6e75"
CPI = "b41fd780-4b6f-43bd-87be-bde431cf6b5a"
WHOLESALE_NOM = "1d2d555c-44c0-4a1d-8ee1-7329df1ec883"
WHOLESALE_REAL = "df9f56e5-f1d9-4d90-a281-9aed3bff1b0f"
LOANS = "fdf0811d-7058-47d2-98ac-286478819d16"
DEBTS = "6f4a156c-4c4f-45a6-a83d-e57db5afb5c5"
ENERGY_PRICES = "94b00f19-3b93-4add-b947-583ec48cd41e"
ANIMAL_PPI = "adc4f2f4-67e7-4fe3-a660-229d38715b75"
ANIMAL_PRICES = "7b15e956-1bd2-4bfd-80e4-5bbf7bc9f910"
INV_PRICES = "ef10704a-6ab1-42cf-a0bd-c0fb2bf438ad"
FREIGHT_TARIFFS = "792a6f5b-f4ef-4233-98db-8da8a7f2a492"
EMPLOYMENT = "2c134407-9399-4674-b131-8c15bd8b1409"
WAGES = "9c2cdc12-ba9d-4d61-b3fe-18c30f63dabd"
BRENT = "45a5dd35-c074-42df-8927-935b81cbfb2b"
NEER = "cccb6b13-dfe0-48a2-b104-b014530ce0d4"
REER = "809c576f-0993-4c3f-a159-06eeab9202f7"
MIACR = "8ea052fc-6455-4d56-8933-5c2cb4efbda9"
M0 = "8aa737a8-1f02-465d-b64b-a60b2a077873"
M1 = "5e3999c8-9303-43e1-8589-fd16dbf6a7c9"
M2 = "79b319c1-4c3b-4645-b246-96423ebb4234"
TRADE = "12a72c2a-4fe8-4d67-997e-333be253406d"

MONTHLY = "ежемесячные данные"
RUB_BN = 217


def _m(name: str, uuid: str, *dims: str, unit_id: int | None = None, kind: Kind = "level", **kw) -> SeriesSpec:
    return SeriesSpec(name, "M", uuid, tuple(dims), unit_id, kind, **kw)


# ---------------------------------------------------------------- quarterly
_GDP_COMPONENTS = {
    "gdp": "Реальный ВВП",
    "cons": "Реальное потребление домохозяйств",
    "inv": "Реальное валовое накопление",
    "inv_cap": "Реальное валовое накопление основного капитала",
}
# newest base first: older bases only extend the history backwards
GDP_BASE_YEARS = (2021, 2008, 2003)

QUARTERLY_SERIES: list[SeriesSpec] = [
    SeriesSpec(
        name=f"{target}_{year}",
        frequency="Q",
        dataset_uuid=GDP,
        dims=(f"в ценах {year} года", component),
        unit_id=RUB_BN,
        kind="splice",
        target=target,
    )
    for target, component in _GDP_COMPONENTS.items()
    for year in GDP_BASE_YEARS
]

# ------------------------------------------------------------------ monthly
_SECTORS = {
    "ipi_mining": "Добыча полезных ископаемых",
    "ipi_manufacturing": "Обрабатывающие производства",
}
_LIVESTOCK = {
    "pv_meat": ("Скот и птица на убой в живом весе", "тыс. тонн.", 224),
    "pv_milk": ("Молоко", "тыс. тонн.", 224),
    "pv_eggs": ("Яйца", "млн. штук", 231),
}
_RETAIL_GROUPS = {
    "": "Всего",
    "_food": "пищевые продукты, включая напитки, и табачные изделия",
    "_nonfood": "непродовольственные товары",
}
_CPI = {
    "cpi": "Индекс потребительских цен",
    "cpi_food": "продукты питания",
    "cpi_alchohol": "алкогольные напитки",
    "cpi_nonfood": "непродовольственные товары",
    "cpi_services": "услуги",
}
_DEBTS = {
    "credit_debt": ("кредиторская задолженность", "всего"),
    "credit_debt_overdue": ("кредиторская задолженность", "просроченная задолженность"),
    "credit_debt_overdue_suppliers": ("кредиторская задолженность", "просроченная задолженность поставщикам"),
    "credit_debt_overdue_gov": ("кредиторская задолженность", "просроченная задолженность в бюджеты всех уровней"),
    "debit_debt": ("дебиторская задолженность", "всего"),
    "debit_debt_overdue": ("дебиторская задолженность", "просроченная задолженность"),
    "debit_debt_overdue_customers": ("дебиторская задолженность", "просроченная задолженность покупателей"),
}
_ENERGY = {
    "ppi_oil": ("рублей за тонну", "Нефть обезвоженная, обессоленная и стабилизированная"),
    "ppi_coal": ("рублей за тонну", "Уголь, за исключением антрацита, угля коксующегося и угля бурого"),
    "ppi_gas": ("рублей за тыс.куб.м", "Газ горючий природный (газ естественный)"),
    "ppi_gasoline": ("рублей за тонну", "Бензин автомобильный"),
    "ppi_diesel": ("рублей за тонну", "Топливо дизельное"),
    "ppi_residual": ("рублей за тонну", "Мазут топочный"),
}
_ANIMAL_PRICES = {
    "ppi_meat": ("рублей за тонну", "Скот крупный рогатый (в живом весе)"),
    "ppi_poultry": ("рублей за тонну", "Птица сельскохозяйственная (в живом весе)"),
    "ppi_milk": ("рублей за тонну", "Молоко сырое крупного рогатого скота"),
    "ppi_eggs": ("рублей за тыс. штук", "Яйца куриные в скорлупе свежие"),
}

# Not collected (too short on IEP): ipi_water_electro_gas (OKVED2, from 2014),
# wage_debt_overdue (from 2013).
MONTHLY_SERIES: list[SeriesSpec] = [
    # industry
    _m("ipi", IPI, MONTHLY, "в % к предыдущему периоду", unit_id=1, kind="chain_mom"),
    *[
        _m(name, IPI_SECTORS, MONTHLY, "отчетный месяц в % к предыдущему месяцу", sector, unit_id=1, kind="chain_mom")
        for name, sector in _SECTORS.items()
    ],
    # agriculture: IEP only publishes % to the same month of previous year
    _m("pi_agriculture", AGRICULTURE, MONTHLY, unit_id=1, kind="chain_yoy"),
    *[
        _m(name, LIVESTOCK, product, MONTHLY, unit, unit_id=unit_id)
        for name, (product, unit, unit_id) in _LIVESTOCK.items()
    ],
    # transport & construction
    _m("comm_freight", FREIGHT, MONTHLY, "млрд тонно-км", unit_id=15),
    _m("rail_load", RAIL, MONTHLY, "млн. тонн", unit_id=223),
    _m("constr_vol", CONSTRUCTION, MONTHLY, "млрд. рублей", unit_id=RUB_BN),
    _m("build_comis", HOUSING, MONTHLY, "млн кв.м общей площади жилых помещений", unit_id=1141),
    # trade & services
    *[
        _m(f"retail_nom{suffix}", RETAIL, group, MONTHLY, "млрд. рублей", unit_id=RUB_BN)
        for suffix, group in _RETAIL_GROUPS.items()
    ],
    *[
        _m(
            f"retail_real{suffix}",
            RETAIL,
            group,
            MONTHLY,
            "в % к предыдущему периоду (в сопоставимых ценах)",
            unit_id=1,
            kind="chain_mom",
        )
        for suffix, group in _RETAIL_GROUPS.items()
    ],
    _m(
        "market_service_vol_real",
        SERVICES,
        MONTHLY,
        "в % к предыдущему периоду (в сопоставимых ценах)",
        unit_id=1328,
        kind="chain_mom",
    ),
    _m("wholesale_nom", WHOLESALE_NOM, MONTHLY, unit_id=RUB_BN),
    # same as the old data: % to the same month of previous year (unit 1331)
    _m("wholesale_real", WHOLESALE_REAL, MONTHLY, unit_id=1331, kind="chain_yoy"),
    # prices
    *[_m(name, CPI, MONTHLY, item, unit_id=1, kind="chain_mom") for name, item in _CPI.items()],
    *[_m(name, ENERGY_PRICES, MONTHLY, unit, product, unit_id=4) for name, (unit, product) in _ENERGY.items()],
    _m("ppi_animal_husbandry", ANIMAL_PPI, MONTHLY, unit_id=1, kind="chain_mom"),
    *[_m(name, ANIMAL_PRICES, MONTHLY, unit, product, unit_id=4) for name, (unit, product) in _ANIMAL_PRICES.items()],
    _m(
        "pi_inv",
        INV_PRICES,
        "Сводный индекс цен на продукцию (затраты, услуги) инвестиционного назначения",
        MONTHLY,
        unit_id=1,
        kind="chain_mom",
    ),
    _m(
        "pi_inv_construction",
        INV_PRICES,
        "в том числе индекс цен производителей на строительную продукцию",
        MONTHLY,
        unit_id=1,
        kind="chain_mom",
    ),
    _m("pi_freight", FREIGHT_TARIFFS, MONTHLY, unit_id=1, kind="chain_mom"),
    # finance
    _m("loans_deposits", LOANS, MONTHLY, unit_id=RUB_BN),
    *[_m(name, DEBTS, side, "млрд. рублей", item, unit_id=RUB_BN) for name, (side, item) in _DEBTS.items()],
    # labour market
    # Rosstat switched from 15-72 to 15+ in 2017; both agree on the 2017 overlap
    _m(
        "un_15plus",
        EMPLOYMENT,
        MONTHLY,
        "%",
        "Уровень безработицы в возрасте 15 лет и старше",
        unit_id=1,
        kind="splice",
        target="un",
    ),
    _m(
        "un_15_72",
        EMPLOYMENT,
        MONTHLY,
        "%",
        "Уровень безработицы в возрасте 15-72 лет",
        unit_id=1,
        kind="splice",
        target="un",
    ),
    _m(
        "labor_demand",
        EMPLOYMENT,
        MONTHLY,
        "тыс. человек",
        "Потребность работодателей в работниках, заявленная в органы службы занятости населения",
        unit_id=221,
    ),
    _m(
        "un_workload",
        EMPLOYMENT,
        MONTHLY,
        "человек",
        "Нагрузка не занятого трудовой деятельностью населения на 100 заявленных вакансий",
        unit_id=108,
    ),
    _m(
        "wage_nom",
        WAGES,
        "Среднемесячная номинальная начисленная заработная плата работников организаций",
        MONTHLY,
        "рублей",
        unit_id=4,
    ),
    _m(
        "wage_real",
        WAGES,
        "Реальная начисленная заработная плата работников организаций",
        MONTHLY,
        "в % к предыдущему периоду",
        unit_id=1,
        kind="chain_mom",
    ),
    # external & monetary (former fred_data / cbr_data)
    _m("brent", BRENT, unit_id=389, search_query="Сырая нефть (Brent)"),
    _m("neer", NEER, unit_id=1555),
    _m("reer", REER, unit_id=1555),
    _m("m0", M0, unit_id=RUB_BN, search_query="Денежный агрегат М0"),
    _m("m1", M1, unit_id=RUB_BN, search_query="Денежный агрегат М1"),
    _m("m2", M2, unit_id=RUB_BN, search_query="Денежный агрегат М2"),
    _m("ex_usd", TRADE, "ежемесячные", "Всего", "Экспорт товаров", unit_id=232),
    _m("im_usd", TRADE, "ежемесячные", "Всего", "Импорт товаров", unit_id=232),
]

# -------------------------------------------------------------------- daily
DAILY_SERIES: list[SeriesSpec] = [
    SeriesSpec("miacr", "D", MIACR, unit_id=1),
]

ALL_SERIES: dict[Frequency, list[SeriesSpec]] = {
    "Q": QUARTERLY_SERIES,
    "M": MONTHLY_SERIES,
    "D": DAILY_SERIES,
}
