import os

import polars

from eng_ru_names_dict import (eng_ru_cbr_dict, eng_ru_fred_dict,
                               eng_ru_gigadata_dict, eng_ru_quarterly_dict)
from preprocess_data.prepare_data import FeaturesService


def run_download_prepared_raw_data() -> None:
    features_service = FeaturesService()
    monthly_data, quarterly_data = features_service.get_features()

    quarterly_data_prep = quarterly_data.select(
        ["dateq"] + list(eng_ru_quarterly_dict.keys())
    )

    monthly_data_prep = monthly_data.select(
        ["datem"]
        + list(eng_ru_gigadata_dict.keys())
        + list(eng_ru_fred_dict.keys())
        + list(eng_ru_cbr_dict.keys())
    )

    os.makedirs("prepared_raw_data", exist_ok=True)

    quarterly_data_prep.write_csv("prepared_raw_data/quarterly_data.csv")
    monthly_data_prep.write_csv("prepared_raw_data/monthly_data.csv")


if __name__ == "__main__":
    run_download_prepared_raw_data()
