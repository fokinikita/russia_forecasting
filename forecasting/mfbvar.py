"""Mixed-frequency Bayesian VAR (R package mfbvar) for all targets and horizons at once.

The model is estimated in R (forecasting/r/mfbvar_forecast.R), Python prepares the data, runs Rscript
and turns its output into the common forecast table, so backtest metrics, spec selection and future
forecasts are computed exactly like for the other models.

    * endogenous series: monthly features (log d12 or their 3-month rolling mean) + 4 quarterly targets
      (log d4, intra-quarterly average aggregation);
    * real time: at origin T with availability a the model sees monthly data up to month a of quarter T
      and targets up to T - 1; one MCMC estimation per origin;
    * failures: every origin is tried `max_attempts` times with different seeds; if it still fails,
      the forecast made at the previous origin is used (up to `max_fallback` quarters back, `refit_date`
      then shows the origin whose estimate was used). R crashes and hangs are handled the same way.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import attrs
import polars as pl

import config
from forecasting.dataset import QuarterlyDataset, shift_quarters

logger = logging.getLogger(__name__)

R_SCRIPT = Path(__file__).resolve().parent / "r" / "mfbvar_forecast.R"
FORECAST_COLUMNS = {"origin": pl.Date, "target": pl.Utf8, "target_date": pl.Date, "pred": pl.Float64}


class MfbvarError(RuntimeError):
    pass


@attrs.frozen
class MfbvarSettings:
    n_lags: int = config.RT_MFBVAR_LAGS
    n_reps: int = config.RT_MFBVAR_N_REPS
    n_burnin: int = config.RT_MFBVAR_N_BURNIN
    prior: str = config.RT_MFBVAR_PRIOR
    variance: str = config.RT_MFBVAR_VARIANCE
    prior_Pi_AR1: float = config.RT_MFBVAR_PRIOR_PI_AR1
    lambda1: float = config.RT_MFBVAR_LAMBDA1
    lambda2: float = config.RT_MFBVAR_LAMBDA2
    lambda3: float = config.RT_MFBVAR_LAMBDA3
    lambda4: float = config.RT_MFBVAR_LAMBDA4
    pred_bands: float = config.RT_MFBVAR_PRED_BANDS
    check_roots: bool = True
    rolling_window: int = config.RT_MFBVAR_ROLLING_WINDOW
    max_attempts: int = config.RT_MFBVAR_MAX_ATTEMPTS
    max_fallback: int = config.RT_MFBVAR_MAX_FALLBACK
    origin_timeout: int = config.RT_MFBVAR_ORIGIN_TIMEOUT
    max_late_start_years: int = config.RT_MFBVAR_MAX_LATE_START_YEARS
    max_variables: int = config.RT_MFBVAR_MAX_VARIABLES
    max_abs_corr: float = 0.95  # near-duplicate series are skipped when the set is capped
    max_staleness_months: int = 6  # monthly series not updated for longer are dropped at that origin
    min_observations: int = 24
    seed: int = config.RANDOM_SEED
    rscript: str = config.RT_RSCRIPT
    script: Path = R_SCRIPT

    def r_options(self) -> dict:
        names = (
            "n_lags", "n_reps", "n_burnin", "prior", "variance", "prior_Pi_AR1", "lambda1", "lambda2",
            "lambda3", "lambda4", "pred_bands", "check_roots", "max_attempts", "max_staleness_months",
            "min_observations", "seed",
        )  # fmt: skip
        return {name: getattr(self, name) for name in names}


# ------------------------------------------------------------------ inputs
@attrs.define(slots=True)
class MfbvarInputs:
    """Monthly and quarterly tables for one specification, in the layout the R script expects."""

    dataset: QuarterlyDataset
    variables: list[str]
    features_type: str
    train_start: int
    settings: MfbvarSettings = attrs.field(factory=MfbvarSettings)
    selection_until: Optional[date] = None  # last quarter whose data may be used to choose variables
    dropped: list[str] = attrs.field(factory=list)

    def _base_columns(self) -> list[str]:
        """'brent_log_d12_m1' -> 'brent_log_d12'; rolling features: only the configured window."""
        columns = [c.removesuffix("_m1") for c in self.dataset.feature_columns(self.features_type, 1, self.variables)]
        if self.features_type == "rolling":
            columns = [c for c in columns if c.endswith(f"_roll_mean_{self.settings.rolling_window}")]
        return columns

    def monthly_long(self) -> pl.DataFrame:
        """Quarterly frame with _m1/_m2/_m3 columns -> one row per month."""
        bases = self._base_columns()
        parts = [
            self.dataset.frame.select(
                pl.col("date").dt.offset_by(f"{month - 1}mo").alias("date"),
                *[pl.col(f"{base}_m{month}").alias(base) for base in bases],
            )
            for month in (1, 2, 3)
        ]
        return pl.concat(parts).sort("date")

    def build(self) -> tuple[pl.DataFrame, pl.DataFrame, date]:
        """(monthly, quarterly, sample_start). All kept monthly series start at `sample_start`."""
        monthly = self.monthly_long()
        latest_start = date(self.train_start + self.settings.max_late_start_years, 1, 1)
        first_obs = {}
        for column in monthly.columns[1:]:
            observed = monthly.filter(pl.col(column).is_not_null())["date"]
            if observed.len():
                first_obs[column] = observed.min()
        keep = [c for c, first in first_obs.items() if first <= latest_start]
        self.dropped = sorted(set(monthly.columns[1:]) - set(keep))
        if not keep:
            raise MfbvarError(f"no monthly series start before {latest_start}")
        if len(keep) > self.settings.max_variables:
            chosen = self._most_informative(keep)
            self.dropped += sorted(set(keep) - set(chosen))
            keep = chosen
        start = max(date(self.train_start, 1, 1), *(first_obs[c] for c in keep))
        # quarterly series are placed in the last month of the quarter -> start at a quarter start
        quarter_start = date(start.year, 3 * ((start.month - 1) // 3) + 1, 1)
        start = quarter_start if quarter_start == start else shift_quarters(quarter_start, 1)
        monthly = monthly.select("date", *keep).filter(pl.col("date") >= start)
        quarterly = self.dataset.frame.select("date", *self.dataset.target_columns).filter(pl.col("date") >= start)
        return monthly, quarterly, start

    def _most_informative(self, columns: list[str]) -> list[str]:
        """Large sets are too slow / singular for MF-BVAR: keep the series most correlated with the targets.

        Correlation of the quarterly average of a monthly series with every target (mean |corr| over targets),
        computed only on quarters up to `selection_until`, so the backtest does not look ahead.
        A series almost identical to an already chosen one is skipped.
        """
        frame = self.dataset.frame.filter(pl.col("date").dt.year() >= self.train_start)
        if self.selection_until is not None:
            frame = frame.filter(pl.col("date") <= self.selection_until)
        quarterly = frame.select(
            *self.dataset.target_columns,
            *[pl.mean_horizontal(f"{c}_m1", f"{c}_m2", f"{c}_m3").alias(c) for c in columns],
        ).to_pandas()
        corr = quarterly.corr(min_periods=12).abs()
        score = corr.loc[columns, self.dataset.target_columns].mean(axis=1).fillna(0).sort_values(ascending=False)
        chosen: list[str] = []
        for column in score.index:
            if all(not (corr.loc[column, other] > self.settings.max_abs_corr) for other in chosen):
                chosen.append(column)
            if len(chosen) == self.settings.max_variables:
                break
        return chosen


# ------------------------------------------------------------------ R call
@attrs.define(slots=True)
class RScriptRunner:
    """Runs the R script for a task directory; kills it when no origin finishes within `origin_timeout`."""

    rscript: str
    script: Path
    origin_timeout: int
    poll_seconds: float = 2.0

    def check(self) -> None:
        if shutil.which(self.rscript) is None and not Path(self.rscript).exists():
            raise MfbvarError(f"Rscript not found ({self.rscript!r}), set the RSCRIPT environment variable")

    @staticmethod
    def _progress(status_path: Path) -> int:
        if not status_path.exists():
            return 0
        with status_path.open(encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)

    def run(self, task_dir: Path) -> Optional[str]:
        """Returns None if R finished normally, otherwise a short description of what went wrong."""
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        log_path = task_dir / "r.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [self.rscript, "--no-save", "--no-restore", str(self.script), str(task_dir)], stdout=log, stderr=log, env=env
            )
            done, last_change = 0, time.monotonic()
            while process.poll() is None:
                time.sleep(self.poll_seconds)
                progress = self._progress(task_dir / "status.csv")
                if progress != done:
                    done, last_change = progress, time.monotonic()
                elif time.monotonic() - last_change > self.origin_timeout:
                    process.kill()
                    process.wait()
                    return f"R killed: no progress for {self.origin_timeout}s"
        if process.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            return f"R exited with code {process.returncode}: {' | '.join(tail)}"
        return None


# ------------------------------------------------------------- forecaster
@attrs.define(slots=True)
class MfbvarForecaster:
    dataset: QuarterlyDataset
    variables: list[str]
    features_type: str
    train_start: int
    availability: int
    horizon: int = config.RT_HORIZON
    settings: MfbvarSettings = attrs.field(factory=MfbvarSettings)
    selection_until: Optional[date] = None  # see MfbvarInputs
    estimates: dict = attrs.field(factory=dict)  # estimation origin -> forecasts frame
    failures: dict = attrs.field(factory=dict)  # estimation origin -> error message
    _inputs: Optional[tuple] = None

    name = "mfbvar"

    @property
    def runner(self) -> RScriptRunner:
        return RScriptRunner(self.settings.rscript, self.settings.script, self.settings.origin_timeout)

    def _prepared(self) -> tuple[pl.DataFrame, pl.DataFrame, date]:
        if self._inputs is None:
            inputs = MfbvarInputs(
                self.dataset, self.variables, self.features_type, self.train_start, self.settings, self.selection_until
            )
            self._inputs = inputs.build()
        return self._inputs

    def _task(self, origins: list[date], sample_start: date) -> dict:
        # enough monthly steps to reach the last horizon even from the deepest fallback origin
        n_fcst_quarters = self.horizon + self.settings.max_fallback
        return {
            **self.settings.r_options(),
            "origins": [o.isoformat() for o in origins],
            "availability": self.availability,
            "sample_start": sample_start.isoformat(),
            "targets": self.dataset.target_columns,
            "n_fcst_months": 3 * n_fcst_quarters,
        }

    def _run_batch(self, origins: list[date]) -> None:
        monthly, quarterly, sample_start = self._prepared()
        with tempfile.TemporaryDirectory(prefix="mfbvar_") as tmp:
            task_dir = Path(tmp)
            monthly.write_csv(task_dir / "monthly.csv")
            quarterly.write_csv(task_dir / "quarterly.csv")
            (task_dir / "task.json").write_text(json.dumps(self._task(origins, sample_start)), encoding="utf-8")
            problem = self.runner.run(task_dir)
            status = self._read(task_dir / "status.csv", {"origin": pl.Date, "status": pl.Utf8, "message": pl.Utf8})
            forecasts = self._read(task_dir / "forecasts.csv", FORECAST_COLUMNS)

        for row in status.iter_rows(named=True):
            if row["status"] == "ok":
                self.estimates[row["origin"]] = forecasts.filter(pl.col("origin") == row["origin"])
            else:
                self.failures[row["origin"]] = row["message"] or "unknown error"
        finished = set(status["origin"].to_list()) if status.height else set()
        rest = [o for o in origins if o not in finished]
        if rest:
            # R died or hung on the first unfinished origin: record it and continue with the others
            self.failures[rest[0]] = problem or "R stopped without a result"
            if rest[1:]:
                self._run_batch(rest[1:])

    @staticmethod
    def _read(path: Path, schema: dict) -> pl.DataFrame:
        if not path.exists():
            return pl.DataFrame(schema=schema)
        frame = pl.read_csv(path, infer_schema_length=0)
        return frame.select([pl.col(name).str.to_date() if dtype == pl.Date else pl.col(name).cast(dtype) for name, dtype in schema.items()])

    def estimate(self, origins: Iterable[date]) -> None:
        todo = sorted({o for o in origins if o not in self.estimates and o not in self.failures})
        if todo:
            self._run_batch(todo)

    def _source(self, origin: date) -> Optional[date]:
        """Origin whose estimate serves `origin`: itself, or the closest earlier successful one."""
        for back in range(self.settings.max_fallback + 1):
            candidate = shift_quarters(origin, -back)
            if candidate in self.estimates:
                return candidate
        return None

    def forecast(self, origins: Iterable[date]) -> pl.DataFrame:
        """target | horizon | origin | target_date | refit_date | pred for every origin (with fallback)."""
        self.runner.check()
        origins = sorted(set(origins))
        self.estimate(origins)
        for back in range(1, self.settings.max_fallback + 1):
            missing = [o for o in origins if self._source(o) is None]
            if not missing:
                break
            self.estimate(shift_quarters(o, -back) for o in missing)

        frames = []
        for origin in origins:
            source = self._source(origin)
            if source is None:
                continue
            last = shift_quarters(origin, self.horizon - 1)
            frames.append(
                self.estimates[source]
                .filter((pl.col("target_date") >= origin) & (pl.col("target_date") <= last))
                .with_columns(
                    pl.lit(origin).alias("origin"),
                    pl.lit(source).alias("refit_date"),
                    pl.col("target").str.replace(r"_log_d4$", ""),
                )
            )
        if not frames:
            first_error = next(iter(self.failures.values()), "no forecasts")
            raise MfbvarError(f"all origins failed, e.g.: {first_error}")
        result = pl.concat(frames)
        # horizon = quarters from the origin to the target quarter + 1
        months = (pl.col("target_date").dt.year() - pl.col("origin").dt.year()) * 12 + (
            pl.col("target_date").dt.month() - pl.col("origin").dt.month()
        )
        return result.with_columns((months // 3 + 1).cast(pl.Int64).alias("horizon"))

    @property
    def fallbacks(self) -> int:
        return len(self.failures)

