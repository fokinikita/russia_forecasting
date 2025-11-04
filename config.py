from typing import Final

# forecast params
HORIZON: Final[int] = 6
VALID_LEN: Final[int] = 12
TEST_LEN: Final[int] = 12

# Catboost params
RANDOM_SEED: Final[int] = 228
THREAD_COUNT: Final[int] = 6
START_ITERATIONS: Final[int] = 1000

# NGB params
START_ITERATIONS_NGB: Final[int] = 500
LEARNING_RATE_NGB: Final[float] = 0.01
DEPTH_NGB_BASE: Final[int] = 2

# features parans
ROLLING_WINDOWS_MONTH: Final[list[int]] = [3, 6, 12]

START_YEAR: Final[int] = 2001

#YEARS_DUMMY: Final[list[int]] = [2008, 2015, 2022]

# paths
GIGA_DATA_PATH: Final[str] = "giga_data_ready_07_2025.xlsx"
FRED_DATA_PATH: Final[str] = "fred_data.xlsx"
CBR_DATA_PATH: Final[str] = "cbr_data.xlsx"
QUARTERLY_DATA_PATH: Final[str] = "quarterly_data.xlsx"
