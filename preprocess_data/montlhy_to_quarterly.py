import polars as pl
import attrs

@attrs.define(slots=True)
class MonthlyToQuarterlyService:
    columns_d12: list
    columns_rolling: list
    avail_features_full: dict = {}

    def _split_by_columns(self, monthly_data: pl.DataFrame, columns_to_fit: list) -> tuple[pl.DataFrame, dict]:

        avail_features = {}

        monthly_data_selected_columns = monthly_data.with_columns([
            ((pl.col("datem").dt.month() - 1) % 3 + 1).alias("month_in_quarter")
        ]).select(
            ['datem', "month_in_quarter"] + columns_to_fit
        )

        first_month_data = monthly_data_selected_columns.filter(
            pl.col("month_in_quarter") == 1
        ).rename(
            {column: column + '_m1' for column in monthly_data_selected_columns.columns if column != 'datem' and column != "month_in_quarter"}
        ).drop(
            "month_in_quarter"
        ).with_columns(
            pl.col('datem').dt.year().alias('year'),
            pl.col('datem').dt.quarter().alias('quarter'),
        )
        avail_features[1] = first_month_data.drop(
            [
                'datem',
                'year',
                'quarter'
            ]
        ).columns

        second_month_data = monthly_data_selected_columns.filter(
            pl.col("month_in_quarter") == 2
        ).rename(
            {column: column + '_m2' for column in monthly_data_selected_columns.columns if column != 'datem' and column != "month_in_quarter"}
        ).with_columns(
            pl.col('datem').dt.year().alias('year'),
            pl.col('datem').dt.quarter().alias('quarter'),
        ).drop(
            "month_in_quarter", 'datem'
        )
        avail_features[2]  = second_month_data.drop(
            [
                'year',
                'quarter'
            ]
        ).columns

        third_month_data = monthly_data_selected_columns.filter(
            pl.col("month_in_quarter") == 3
        ).rename(
            {column: column + '_m3' for column in monthly_data_selected_columns.columns if column != 'datem' and column != "month_in_quarter"}
        ).with_columns(
            pl.col('datem').dt.year().alias('year'),
            pl.col('datem').dt.quarter().alias('quarter'),
        ).drop(
            "month_in_quarter", 'datem'
        )
        avail_features[3]  = third_month_data.drop(
            [
                'year',
                'quarter'
            ]
        ).columns

        monthly_data_quarterly_split = first_month_data.join(
            second_month_data, on=['year', 'quarter'],how='left'
        ).join(
            third_month_data, on=['year', 'quarter'],how='left'
        )

        return monthly_data_quarterly_split, avail_features

    def _split_monthly_to_quarterly(self, monthly_data: pl.DataFrame) -> dict:

        self.avail_features_full = {"rolling": {}, "d12": {}}

        rolling_features, avail_features_rolling = self._split_by_columns(monthly_data, self.columns_rolling)
        self.avail_features_full["rolling"] = avail_features_rolling
        d12_features, avail_features_d12 = self._split_by_columns(monthly_data, self.columns_d12)
        self.avail_features_full["d12"] = avail_features_d12

        features_mtoq = pl.concat([d12_features, rolling_features.drop(['datem', 'year', 'quarter'])], how="horizontal")

        return features_mtoq

    def _join_quarterly_data(self, features_mtoq: pl.DataFrame, quarterly_data: pl.DataFrame) -> pl.DataFrame:
        joined = quarterly_data.rename(
            {
                'dateq': 'date'
            }
        ).join(
            features_mtoq.rename(
                {
                    'datem': 'date'
                }
            ), on='date')

        return joined

    def run_transorm(self, monthly_data: pl.DataFrame, quarterly_data: pl.DataFrame) -> pl.DataFrame:
        features_mtoq = self._split_monthly_to_quarterly(monthly_data)
        features = self._join_quarterly_data(features_mtoq, quarterly_data)

        features = features.drop(['year', 'quarter'])
        features = features.select(['date'] + [col for col in features.columns if col != 'date'])
        return features
