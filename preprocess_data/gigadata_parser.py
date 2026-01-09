import logging

import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_giga_data(filename: str) -> list[pl.DataFrame]:
    xls = pd.ExcelFile(filename)
    sheet_names = xls.sheet_names

    parsed_result = []

    for sheet_name in sheet_names:
        try:
            df_raw = pd.read_excel(filename, header=None, sheet_name=sheet_name)

            block_starts = df_raw[
                df_raw[0].apply(lambda x: isinstance(x, str))
                & df_raw.iloc[:, 1:].isna().all(axis=1)
            ].index.tolist()

            all_blocks = []

            for i, start in enumerate(block_starts):
                name = str(df_raw.iloc[start, 0]).strip()
                end = block_starts[i + 1] if i + 1 < len(block_starts) else len(df_raw)

                block = df_raw.iloc[start + 1 : end].copy()

                # удаляем пустые строки
                block = block.dropna(how="all")
                if block.empty:
                    continue

                # переименовываем колонки — первая колонка это "year"
                block.columns = range(block.shape[1])
                block = block.rename(columns={0: "year"})

                # оставляем только числовые месяцы (1..12)
                month_cols = [c for c in range(1, 13) if c in block.columns]
                if not month_cols:
                    continue

                # “расплавляем” в длинный формат
                block_long = block.melt(
                    id_vars="year",
                    value_vars=month_cols,
                    var_name="month",
                    value_name="value",
                )

                # чистим от NaN и неправильных годов
                block_long = block_long.dropna(subset=["year", "value"])
                block_long = block_long[
                    block_long["year"].apply(lambda x: str(x).isdigit())
                ]

                # === ЧИСТИМ значения ===
                block_long["value"] = (
                    block_long["value"]
                    .astype(str)
                    .str.replace(
                        r"[^0-9,.\-]", "", regex=True
                    )  # убираем скобки, символы, сноски
                    .str.replace(",", ".", regex=False)  # запятые → точки
                )

                # Преобразуем в float (невалидные → NaN)
                block_long["value"] = pd.to_numeric(
                    block_long["value"], errors="coerce"
                )

                # Удаляем пустые значения после очистки
                block_long = block_long.dropna(subset=["value"])
                # создаём дату
                block_long["year"] = block_long["year"].astype(int)
                block_long["month"] = block_long["month"].astype(int)
                block_long["date"] = pd.to_datetime(
                    block_long["year"].astype(str)
                    + "-"
                    + block_long["month"].astype(str)
                    + "-01"
                )

                block_long["indicator"] = name

                all_blocks.append(block_long[["date", "indicator", "value"]])

                result_long = pd.concat(all_blocks, ignore_index=True)
                parsed_list = pd.pivot(
                    result_long, index="date", columns="indicator", values="value"
                )

                parsed_list_polars = pl.DataFrame(
                    parsed_list.reset_index()
                ).with_columns(pl.col("date").cast(pl.Date).alias("date"))

            parsed_result.append(parsed_list_polars)
            # logger.info(f'List {sheet_name} was succesfully parsed')
        except Exception as e:
            pass
            # logger.info(f'List {sheet_name} was not parsed')
            # logger.error(f'ERROR message: {e}')

    return parsed_result
