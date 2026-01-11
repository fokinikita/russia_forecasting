from typing import Final

# Common params
RANDOM_SEED: Final[int] = 228
THREAD_COUNT: Final[int] = 4

# Forecast params
HORIZON: Final[int] = 6
VALID_LEN: Final[int] = 12
TEST_LEN: Final[int] = 12
MAX_AVALIABILITY: Final[int] = 3

# Catboost params
START_ITERATIONS: Final[int] = 1000
EARLY_STOPPING_ROUNDS: Final[int] = 50

# NGB params
START_ITERATIONS_NGB: Final[int] = 500
LEARNING_RATE_NGB: Final[float] = 0.01
DEPTH_NGB_BASE: Final[int] = 2
EARLY_STOPPING_ROUNDS_NGB: Final[int] = 25

# TabNet params
BATCH_SIZE_TABNET: Final[int] = 8
VIRTUAL_BATCH_SIZE_TABNET: Final[int] = 8
LEARNING_RATE_TABNET: Final[float] = 0.05
LAMBDA_SPARSE_TABNET: Final[float] = 0.005
MASK_TYPE_TABNET: Final[str] = "sparsemax"
MAX_EPOCHS_TABNET: Final[int] = 200
EARLY_STOPPING_ROUNDS_TABNET: Final[int] = 20

# DFM params
K_FACTORS_GRID = [1, 2]
FACTOR_ORDER_GRID = [1]

# features params
ROLLING_WINDOWS_MONTH: Final[list[int]] = [3, 6, 12]

START_YEAR: Final[int] = 2001

# YEARS_DUMMY: Final[list[int]] = [2008, 2015, 2022]

# paths
GIGA_DATA_PATH: Final[str] = "data/giga_data_ready_07_2025.xlsx"
FRED_DATA_PATH: Final[str] = "data/fred_data.xlsx"
CBR_DATA_PATH: Final[str] = "data/cbr_data.xlsx"
QUARTERLY_DATA_PATH: Final[str] = "data/quarterly_data.xlsx"
