from typing import Final, Optional

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

# data source for FeaturesService: "manual" (excel files above) or "iep" (data.iep.ru)
DATA_SOURCE: Final[str] = "manual"
# where data_iep.pipeline saves collected data
IEP_DATA_DIR: Final[str] = "data/iep"

# ---------------------------------------------------------------------------
# Real-time experiments on IEP data (package `forecasting`)
# ---------------------------------------------------------------------------
RT_TARGETS: Final[list[str]] = ["gdp", "cons", "inv", "inv_cap"]
RT_HORIZON: Final[int] = 8  # quarters ahead (2 years), h=1 is the nowcast of the current quarter
RT_TEST_LEN: Final[int] = 12  # last quarters with known targets used as the test period
RT_VALID_LEN: Final[int] = 12  # last quarters of every training sample used for early stopping
RT_REFIT_EVERY: Final[int] = 4  # re-estimate models every N forecast origins (1 = every quarter)
RT_TRAIN_STARTS: Final[list[int]] = [2000, 2010]
RT_FEATURE_TYPES: Final[list[str]] = ["classic", "rolling"]  # classic = log/diff d12, rolling = its rolling means
RT_AVAILABILITIES: Final[list[int]] = [1, 2, 3]  # months of the current quarter already known
RT_VARIABLE_SETS: Final[dict[str, Optional[list[str]]]] = {
    # brent, real retail turnover (physical volume), MIACR, industrial production
    "short": ["brent", "retail_real", "miacr", "ipi"],
    # + REER, exports, imports (FOB, mln USD), unemployment, commercial freight turnover
    "medium": ["brent", "retail_real", "miacr", "ipi", "reer", "ex_usd", "im_usd", "un", "comm_freight"],
    "all": None,  # every collected monthly variable
}
RT_MODELS: Final[list[str]] = ["gb", "ngb", "tabnet", "dfm"]
RT_N_JOBS: Final[int] = 7  # parallel worker processes
RT_DFM_N_JOBS: Final[int] = 3  # DFM on all variables needs ~2-4 GB per process
RT_CONSENSUS_TOP_K: Final[int] = 10  # future consensus = median of the K best backtest specs
RT_DFM_FACTORS: Final[int] = 2
RT_DFM_FACTOR_ORDER: Final[int] = 1
RT_RESULTS_DIR: Final[str] = "results"
