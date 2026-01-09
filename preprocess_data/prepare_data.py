import logging

import attrs
import polars as pl

import config
from eng_ru_names_dict import chain_indeces, eng_ru_quarterly_dict
from preprocess_data.gigadata_parser import parse_giga_data

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@attrs.define(slots=True)
class FeaturesService:

    columns_rolling: list = []
    columns_d12: list = []
    columns_d4: list = []

    @staticmethod
    def _read_monthly_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

        cbr_data_raw = (
            pl.read_excel(config.CBR_DATA_PATH)
            .with_columns(pl.col("date").cast(pl.Date).alias("datem"))
            .drop("date")
        )

        fred_data_raw = (
            pl.read_excel(config.FRED_DATA_PATH)
            .with_columns(pl.col("date").cast(pl.Date).alias("datem"))
            .drop("date")
        )

        quarterly_data_raw = (
            pl.read_excel(config.QUARTERLY_DATA_PATH)
            .with_columns(pl.col("date").cast(pl.Date).alias("dateq"))
            .with_columns(
                pl.col("dateq").dt.quarter().alias("quarter"),
                pl.col("dateq").dt.year().alias("year"),
            )
            .drop("date")
            .sort("dateq")
        )

        return (cbr_data_raw, fred_data_raw, quarterly_data_raw)

    @staticmethod
    def _parse_giga_data() -> pl.DataFrame:
        giga_data_list = parse_giga_data(config.GIGA_DATA_PATH)

        min_date = min(sheet["date"].min() for sheet in giga_data_list)
        max_date = max(sheet["date"].max() for sheet in giga_data_list)

        full_dates = pl.DataFrame(
            {
                "date": pl.date_range(
                    start=min_date, end=max_date, interval="1mo", eager=True
                )
            }
        )

        joined = full_dates

        for i, df in enumerate(giga_data_list, start=1):
            if "date" not in df.columns:
                raise ValueError(f"В таблице {i} нет колонки 'date'")

            joined = joined.join(df, on="date", how="left")

            # logger.info(f"Присоединен лист {i}/{len(giga_data_list)} — {df.columns}")

        joined = joined.sort("date")
        giga_data = joined.rename(
            {
                "date": "datem",
            }
        )

        # logger.info(f" Диапазон: {min_date} → {max_date}")
        # logger.info(f"Размер итогового DataFrame: {joined.shape}")

        return giga_data

    @staticmethod
    def _convert_indeces_to_basics(
        giga_data: pl.DataFrame, chain_indeces: list
    ) -> pl.DataFrame:
        giga_data = giga_data.sort("datem")

        for column in chain_indeces:
            chain_series = giga_data[column]
            first_non_null = chain_series.drop_nulls()[0]

            base_index = []
            last_value = None
            for x in chain_series:
                if x is None:
                    base_index.append(None)
                else:
                    if last_value is None:
                        last_value = first_non_null
                    else:
                        last_value = last_value * x / 100
                    base_index.append(last_value)
            giga_data = giga_data.with_columns(pl.Series(column, base_index))

        return giga_data

    @staticmethod
    def _join_monhtly_data(
        giga_data: pl.DataFrame, fred_data_raw: pl.DataFrame, cbr_data_raw: pl.DataFrame
    ) -> pl.DataFrame:

        monthly_data = giga_data.join(fred_data_raw, how="left", on="datem").join(
            cbr_data_raw, how="left", on="datem"
        )
        return monthly_data

    def _get_features_monthly(self, monthly_data: pl.DataFrame) -> pl.DataFrame:
        monthly_data = monthly_data.sort("datem")

        def get_possibly_log_variables(monthly_data: pl.DataFrame) -> list[str]:
            columns_possibly_log = []
            for column in monthly_data.columns:

                if column != "datem":
                    negative_count = monthly_data.filter(pl.col(column) <= 0).height
                    if negative_count == 0:
                        columns_possibly_log.append(column)

            return columns_possibly_log

        columns_possibly_log = get_possibly_log_variables(monthly_data)
        columns_not_possibly_log = list(
            set(monthly_data.columns) - set(columns_possibly_log)
        )
        columns_not_possibly_log = [
            column for column in columns_not_possibly_log if column != "datem"
        ]

        columns_d12_log = []
        d12_log_expr = []
        for column in columns_possibly_log:
            new_column = column + "_log" + "_d12"
            d12_log_expr.append(
                (pl.col(column).log() - pl.col(column).log().shift(12)).alias(
                    new_column
                )
            )

            columns_d12_log.append(new_column)

        columns_d12_no_log = []
        d12_no_log_expr = []
        for column in columns_not_possibly_log:
            new_column = column + "_d12"
            d12_no_log_expr.append(
                (pl.col(column) - pl.col(column).shift(12)).alias(new_column)
            )
            columns_d12_no_log.append(new_column)

        self.columns_d12 = columns_d12_log + columns_d12_no_log

        monthly_data = monthly_data.with_columns(d12_log_expr).with_columns(
            d12_no_log_expr
        )

        rolling_mean_d12 = []
        columns_rolling_tmp = []
        for roll_window in config.ROLLING_WINDOWS_MONTH:
            for column in self.columns_d12:
                new_column = column + f"_roll_mean_{roll_window}"
                rolling_mean_d12.append(
                    pl.col(column).rolling_mean(roll_window).alias(new_column)
                )
                columns_rolling_tmp.append(new_column)

        self.columns_rolling = columns_rolling_tmp

        monthly_data = monthly_data.with_columns(rolling_mean_d12)

        return monthly_data

    def _prepare_quarterly_data(self, quarterly_data_raw: pl.DataFrame) -> pl.DataFrame:
        quarterly_transform_expr = []
        quarterly_transform = {}

        columns_d4_tmp = []
        for column in eng_ru_quarterly_dict.keys():
            new_column = column + "_log_d4"
            quarterly_transform_expr.append(
                (pl.col(column).log() - pl.col(column).log().shift(4)).alias(new_column)
            )
            quarterly_transform[column] = new_column
            columns_d4_tmp.append(new_column)

        self.columns_d4 = columns_d4_tmp
        quarterly_data = quarterly_data_raw.with_columns(quarterly_transform_expr)
        return quarterly_data

    def get_features(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        cbr_data_raw, fred_data_raw, quarterly_data_raw = self._read_monthly_data()
        giga_data = self._parse_giga_data()
        giga_data = self._convert_indeces_to_basics(giga_data, chain_indeces)
        joined_monthly_data = self._join_monhtly_data(
            giga_data, fred_data_raw, cbr_data_raw
        )
        monthly_features = self._get_features_monthly(joined_monthly_data)

        quarterly_data = self._prepare_quarterly_data(quarterly_data_raw)

        return monthly_features, quarterly_data
