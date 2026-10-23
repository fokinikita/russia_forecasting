import os
from typing import Final, Optional

# Common params
RANDOM_SEED: Final[int] = 228

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

# features params (forecasting/features.py): rolling means of the d12 features
ROLLING_WINDOWS_MONTH: Final[list[int]] = [3, 6, 12]

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
RT_MODELS: Final[list[str]] = ["gb", "ngb", "tabnet", "dfm", "mfbvar"]
RT_N_JOBS: Final[int] = 7  # parallel worker processes
RT_DFM_N_JOBS: Final[int] = 3  # DFM on all variables needs ~2-4 GB per process
RT_CONSENSUS_TOP_K: Final[int] = 10  # future consensus = median of the K best backtest specs
RT_DFM_FACTORS: Final[int] = 2
RT_DFM_FACTOR_ORDER: Final[int] = 1
RT_RESULTS_DIR: Final[str] = "results"

# Automatic choice of the specification (backtest/selected_specs.csv, future/selected.csv):
# for every model x target x availability the train start, variable set (model size) and features type
# with the lowest RMSE relative to naive, averaged over all horizons, are chosen.
RT_SELECTION_METRIC: Final[str] = "rel_rmse_naive"

# Future forecasts that failed in this run are taken from the latest previous run (same spec, overlapping quarters)
RT_FUTURE_FALLBACK_TO_PREVIOUS_RUN: Final[bool] = True

# ---------------------------------------------------------------------------
# MF-BVAR (R package mfbvar 0.5.6, archived on CRAN -> installed from mfbvar_0.5.6.tar.gz)
# Same grid as the other models; one MCMC estimation per forecast origin (refit_every is not used).
# ---------------------------------------------------------------------------
RT_RSCRIPT: Final[str] = os.environ.get("RSCRIPT", "Rscript")  # path to Rscript, e.g. "C:/Program Files/R/R-4.1.0/bin/Rscript.exe"
RT_MFBVAR_N_JOBS: Final[int] = int(os.environ.get("MFBVAR_N_JOBS", RT_N_JOBS))  # parallel R processes (1 core each)
RT_MFBVAR_LAGS: Final[int] = 3  # monthly lags (>= 3 for intra-quarterly averaging)
RT_MFBVAR_N_REPS: Final[int] = 500  # MCMC draws kept
RT_MFBVAR_N_BURNIN: Final[int] = 500
RT_MFBVAR_PRIOR: Final[str] = "minn"  # minn / ss / ssng
RT_MFBVAR_VARIANCE: Final[str] = "iw"  # iw / diffuse / csv / fsv
RT_MFBVAR_PRIOR_PI_AR1: Final[float] = 0.7  # prior mean of own first lag (series are y/y growth rates -> persistent)
RT_MFBVAR_LAMBDA1: Final[float] = 0.1  # overall tightness
RT_MFBVAR_LAMBDA2: Final[float] = 0.5  # cross-variable tightness
RT_MFBVAR_LAMBDA3: Final[float] = 1.0  # lag decay
RT_MFBVAR_LAMBDA4: Final[float] = 100.0  # intercept
RT_MFBVAR_PRED_BANDS: Final[float] = 0.68
RT_MFBVAR_ROLLING_WINDOW: Final[int] = 3  # "rolling" features for MF-BVAR: only this window (3 windows = collinear VAR)
RT_MFBVAR_MAX_ATTEMPTS: Final[int] = 2  # estimation attempts per origin (new seed each time)
RT_MFBVAR_MAX_FALLBACK: Final[int] = 4  # failed origin -> forecast of up to N quarters earlier origin
RT_MFBVAR_ORIGIN_TIMEOUT: Final[int] = 3600  # seconds without progress before the R process is killed
RT_MFBVAR_MAX_VARIABLES: Final[int] = 20  # "all" set is capped: 61 series -> singular / hours per estimate
RT_MFBVAR_MAX_LATE_START_YEARS: Final[int] = 2  # drop variables starting later than train_start + N years
