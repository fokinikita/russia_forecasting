"""Experiment grid, real-time backtest plan and the parallel runner.

Pseudo real-time backtest
-------------------------
Test period = last `test_len` quarters with known targets. Every forecast of a
test quarter t at horizon h is made from origin T = t - (h - 1) using

    * a model estimated only on targets known before T (expanding window,
      re-estimated every `refit_every` origins and reused in between);
    * features of quarter T (month `availability`) and the target of T - 1.

Future forecasts use the same code with a single origin: the quarter after the
last known target, the model being estimated on everything known.

DFM and MF-BVAR are system models: one specification covers all targets and horizons
(MF-BVAR forecasts recursively and is re-estimated at every origin, see forecasting/mfbvar.py).
"""

from __future__ import annotations

import contextlib
import io
import logging
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import attrs
import polars as pl

import config
from forecasting.dataset import QuarterlyDataset, shift_quarters
from forecasting.design import TARGET_LAG, DirectDesign
from forecasting.dfm import DFMForecaster
from forecasting.mfbvar import MfbvarForecaster
from forecasting.models import DIRECT_MODELS, make_direct_model

logger = logging.getLogger(__name__)

SYSTEM_MODELS = ("dfm", "mfbvar")  # one estimate for all targets and horizons
CONFIG_COLUMNS = ["model", "train_start", "var_set", "features_type", "availability"]
FORECAST_SCHEMA = {
    "model": pl.Utf8,
    "train_start": pl.Int64,
    "var_set": pl.Utf8,
    "features_type": pl.Utf8,
    "availability": pl.Int64,
    "target": pl.Utf8,
    "horizon": pl.Int64,
    "origin": pl.Date,
    "target_date": pl.Date,
    "refit_date": pl.Date,
    "pred": pl.Float64,
    "origin_data_share": pl.Float64,
}


# ---------------------------------------------------------------- settings
@attrs.frozen
class ExperimentSettings:
    targets: tuple[str, ...] = tuple(config.RT_TARGETS)
    horizon: int = config.RT_HORIZON
    test_len: int = config.RT_TEST_LEN
    refit_every: int = config.RT_REFIT_EVERY
    train_starts: tuple[int, ...] = tuple(config.RT_TRAIN_STARTS)
    feature_types: tuple[str, ...] = tuple(config.RT_FEATURE_TYPES)
    availabilities: tuple[int, ...] = tuple(config.RT_AVAILABILITIES)
    variable_sets: dict = attrs.field(factory=lambda: dict(config.RT_VARIABLE_SETS))


@attrs.frozen
class ExperimentConfig:
    """One estimated specification. Direct models: one per target & horizon; DFM: all at once."""

    model: str
    train_start: int
    var_set: str
    features_type: str
    availability: int
    target: Optional[str] = None
    horizon: Optional[int] = None

    @property
    def label(self) -> str:
        parts = [self.model, str(self.train_start), self.var_set, self.features_type, f"a{self.availability}"]
        if self.target:
            parts += [self.target, f"h{self.horizon}"]
        return "/".join(parts)

    @classmethod
    def from_label(cls, label: str) -> "ExperimentConfig":
        """Inverse of `label`: 'gb/2000/short/classic/a1/gdp/h3' -> ExperimentConfig."""
        parts = label.split("/")
        model, start, var_set, ftype, avail = parts[:5]
        target, horizon = (parts[5], int(parts[6][1:])) if len(parts) == 7 else (None, None)
        return cls(model, int(start), var_set, ftype, int(avail[1:]), target, horizon)

    def tags(self) -> dict:
        return {
            "model": self.model,
            "train_start": self.train_start,
            "var_set": self.var_set,
            "features_type": self.features_type,
            "availability": self.availability,
        }


@attrs.define(slots=True)
class ExperimentGrid:
    settings: ExperimentSettings

    def configs(self, model: str) -> list[ExperimentConfig]:
        base = [
            (start, var_set, ftype, avail)
            for start in self.settings.train_starts
            for var_set in self.settings.variable_sets
            for ftype in self.settings.feature_types
            for avail in self.settings.availabilities
        ]
        if model in SYSTEM_MODELS:
            return [ExperimentConfig(model, *b) for b in base]
        if model not in DIRECT_MODELS:
            raise ValueError(f"unknown model {model!r}")
        return [
            ExperimentConfig(model, *b, target=target, horizon=h)
            for b in base
            for target in self.settings.targets
            for h in range(1, self.settings.horizon + 1)
        ]


# -------------------------------------------------------------------- plan
@attrs.frozen
class ForecastPlan:
    """Which origins are forecast and when models are re-estimated.

    segments: [(refit_date, known_until, [origins served by this estimate])]
    target_dates: forecasts are kept only for these target quarters (None = all).
    """

    segments: tuple[tuple[date, date, tuple[date, ...]], ...]
    target_dates: Optional[frozenset] = None

    @classmethod
    def backtest(cls, dataset: QuarterlyDataset, settings: ExperimentSettings) -> "ForecastPlan":
        test_dates = dataset.test_dates(settings.test_len)
        first_origin = shift_quarters(test_dates[0], -(settings.horizon - 1))
        origins = []
        current = first_origin
        while current <= test_dates[-1]:
            origins.append(current)
            current = shift_quarters(current, 1)
        segments = []
        for i in range(0, len(origins), settings.refit_every):
            block = tuple(origins[i : i + settings.refit_every])
            segments.append((block[0], shift_quarters(block[0], -1), block))
        return cls(tuple(segments), frozenset(test_dates))

    @classmethod
    def future(cls, dataset: QuarterlyDataset) -> "ForecastPlan":
        last = dataset.last_target_date
        origin = shift_quarters(last, 1)
        return cls(((origin, last, (origin,)),), None)

    def keep(self, target_date: date) -> bool:
        return self.target_dates is None or target_date in self.target_dates


# ------------------------------------------------------------- model sink
@attrs.frozen
class ModelSink:
    """Saves fitted models of chosen specifications (used for the future forecasts).

    keys: "model/train_start/var_set/features_type/a<availability>/target" of the specs to keep
    (backtest/selected_specs.csv). Direct models -> one joblib file per target x horizon; DFM -> its
    estimated parameters (enough to rebuild the state-space model on the same data). MF-BVAR lives in R
    and is re-estimated at every origin, it is not saved.
    """

    directory: Path
    keys: frozenset

    @staticmethod
    def key(cfg: ExperimentConfig, target: str) -> str:
        return "/".join([cfg.model, str(cfg.train_start), cfg.var_set, cfg.features_type, f"a{cfg.availability}", target])

    def wants(self, cfg: ExperimentConfig, target: Optional[str] = None) -> bool:
        targets = [target] if target else [t for t in config.RT_TARGETS]
        return any(self.key(cfg, t) in self.keys for t in targets)

    def _path(self, cfg: ExperimentConfig, suffix: str) -> Path:
        path = self.directory / cfg.model / f"{cfg.label.replace('/', '__')}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_direct(self, cfg: ExperimentConfig, model, feature_cols: tuple[str, ...], known_until: date) -> None:
        import joblib

        try:
            joblib.dump(
                {
                    "config": cfg.label,
                    "target": cfg.target,
                    "horizon": cfg.horizon,
                    "feature_columns": list(feature_cols),
                    "targets_known_until": known_until.isoformat(),
                    "model": model,
                },
                self._path(cfg, ".joblib"),
                compress=3,
            )
        except Exception as error:  # saving a model must never break the forecast
            logger.warning(f"{cfg.label}: model not saved ({error})")

    def save_dfm(self, cfg: ExperimentConfig, dfm: DFMForecaster, known_until: date) -> None:
        try:
            results = dfm.results
            pl.DataFrame({"param": list(results.model.param_names), "value": list(map(float, results.params))}).with_columns(
                pl.lit(cfg.label).alias("config"),
                pl.lit(known_until.isoformat()).alias("targets_known_until"),
                pl.lit(",".join(results.model.endog_names)).alias("endog_names"),
            ).write_parquet(self._path(cfg, ".params.parquet"))
        except Exception as error:
            logger.warning(f"{cfg.label}: DFM parameters not saved ({error})")


# ------------------------------------------------------------------- tasks
def _feature_cols(dataset: QuarterlyDataset, settings: ExperimentSettings, cfg: ExperimentConfig) -> tuple[str, ...]:
    variables = dataset.resolve_variable_set(cfg.var_set, settings.variable_sets[cfg.var_set])
    return tuple(dataset.feature_columns(cfg.features_type, cfg.availability, variables))


def _origin_data_share(dataset: QuarterlyDataset, feature_cols: tuple[str, ...], origin: date) -> float:
    """Share of the features actually published for the origin quarter (before carry-forward)."""
    row = dataset.frame.filter(pl.col("date") == origin).select(feature_cols)
    if row.height == 0 or not feature_cols:
        return 0.0
    return float(sum(v is not None for v in row.row(0)) / len(feature_cols))


def run_direct(
    dataset: QuarterlyDataset,
    settings: ExperimentSettings,
    cfg: ExperimentConfig,
    plan: ForecastPlan,
    sink: Optional[ModelSink] = None,
) -> pl.DataFrame:
    feature_cols = _feature_cols(dataset, settings, cfg)
    design = DirectDesign(dataset, cfg.target, cfg.horizon, feature_cols)
    frame = design.frame()
    rows = []
    for refit_date, known_until, origins in plan.segments:
        train = design.training_rows(frame, cfg.train_start, known_until)
        if train.height < 20:
            logger.warning(f"{cfg.label}: only {train.height} training rows at {refit_date}, skipped")
            continue
        wanted = [o for o in origins if plan.keep(shift_quarters(o, cfg.horizon - 1))]
        to_predict = frame.filter(pl.col("origin").is_in(wanted) & pl.col(TARGET_LAG).is_not_null())
        if to_predict.height == 0:
            continue
        x_train, y_train = design.to_numpy(train)
        x_pred, _ = design.to_numpy(to_predict)
        model = make_direct_model(cfg.model).fit(x_train, y_train)
        preds = model.predict(x_pred)
        if sink is not None and sink.wants(cfg, cfg.target):
            sink.save_direct(cfg, model, feature_cols, known_until)
        for origin, target_date, pred in zip(to_predict["origin"], to_predict["target_date"], preds):
            rows.append(
                {
                    **cfg.tags(),
                    "target": cfg.target,
                    "horizon": cfg.horizon,
                    "origin": origin,
                    "target_date": target_date,
                    "refit_date": refit_date,
                    "pred": float(pred),
                    "origin_data_share": _origin_data_share(dataset, feature_cols, origin),
                }
            )
    return pl.DataFrame(rows, schema=FORECAST_SCHEMA)


def run_dfm(
    dataset: QuarterlyDataset,
    settings: ExperimentSettings,
    cfg: ExperimentConfig,
    plan: ForecastPlan,
    sink: Optional[ModelSink] = None,
) -> pl.DataFrame:
    feature_cols = _feature_cols(dataset, settings, cfg)
    frames = []
    for refit_date, known_until, origins in plan.segments:
        dfm = DFMForecaster(dataset, feature_cols, cfg.train_start).fit(known_until)
        if sink is not None and sink.wants(cfg):
            sink.save_dfm(cfg, dfm, known_until)
        for origin in origins:
            forecast = dfm.forecast(origin, settings.horizon).filter(
                pl.col("target_date").map_elements(plan.keep, return_dtype=pl.Boolean)
            )
            if forecast.height:
                frames.append(
                    forecast.with_columns(
                        *[pl.lit(v).alias(k) for k, v in cfg.tags().items()],
                        pl.lit(refit_date).alias("refit_date"),
                        pl.lit(_origin_data_share(dataset, feature_cols, origin)).alias("origin_data_share"),
                    )
                )
    if not frames:
        return pl.DataFrame(schema=FORECAST_SCHEMA)
    return pl.concat(frames, how="diagonal").select(
        [pl.col(name).cast(dtype) for name, dtype in FORECAST_SCHEMA.items()]
    )


def run_mfbvar(
    dataset: QuarterlyDataset,
    settings: ExperimentSettings,
    cfg: ExperimentConfig,
    plan: ForecastPlan,
    sink: Optional[ModelSink] = None,  # MF-BVAR (R) models are not saved
) -> pl.DataFrame:
    variables = dataset.resolve_variable_set(cfg.var_set, settings.variable_sets[cfg.var_set])
    origins = [origin for _, _, block in plan.segments for origin in block]
    forecaster = MfbvarForecaster(
        dataset,
        variables,
        cfg.features_type,
        cfg.train_start,
        cfg.availability,
        horizon=settings.horizon,
        # variables of capped sets are chosen on data known before the first origin
        selection_until=min(known for _, known, _ in plan.segments),
    )
    forecast = forecaster.forecast(origins).filter(
        pl.col("target_date").map_elements(plan.keep, return_dtype=pl.Boolean)
    )
    if forecast.height == 0:
        return pl.DataFrame(schema=FORECAST_SCHEMA)
    feature_cols = _feature_cols(dataset, settings, cfg)
    shares = {o: _origin_data_share(dataset, feature_cols, o) for o in forecast["origin"].unique().to_list()}
    return forecast.with_columns(
        *[pl.lit(v).alias(k) for k, v in cfg.tags().items()],
        pl.col("origin").replace_strict(shares, return_dtype=pl.Float64).alias("origin_data_share"),
    ).select([pl.col(name).cast(dtype) for name, dtype in FORECAST_SCHEMA.items()])


RUNNERS = {"dfm": run_dfm, "mfbvar": run_mfbvar}


def run_task(
    dataset: QuarterlyDataset,
    settings: ExperimentSettings,
    cfg: ExperimentConfig,
    plan: ForecastPlan,
    sink: Optional[ModelSink] = None,
) -> tuple[ExperimentConfig, Optional[pl.DataFrame], Optional[str]]:
    """Worker entry point: never raises, returns (config, forecasts | None, error | None)."""
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            runner = RUNNERS.get(cfg.model, run_direct)
            return cfg, runner(dataset, settings, cfg, plan, sink), None
    except Exception as error:  # one broken spec must not stop the night run
        return cfg, None, f"{type(error).__name__}: {error}"


# ------------------------------------------------------------------ runner
@attrs.define(slots=True)
class ExperimentRunner:
    dataset: QuarterlyDataset
    settings: ExperimentSettings = attrs.field(factory=ExperimentSettings)
    n_jobs: int = config.RT_N_JOBS
    errors: list = attrs.field(factory=list)
    sink: Optional[ModelSink] = None  # saves fitted models of chosen specs (future forecasts)

    def run(
        self, models: Iterable[str], plan: ForecastPlan, checkpoint_dir: Optional[Path] = None, resume: bool = True
    ) -> pl.DataFrame:
        frames = []
        for model in models:
            checkpoint = checkpoint_dir / f"forecasts_{model}.parquet" if checkpoint_dir else None
            if checkpoint is not None and resume and checkpoint.exists():
                logger.info(f"[{model}] already done, loading {checkpoint}")
                frames.append(pl.read_parquet(checkpoint))
                continue
            result = self.run_model(model, plan)
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                result.write_parquet(checkpoint)
            frames.append(result)
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame(schema=FORECAST_SCHEMA)

    def run_model(
        self, model: str, plan: ForecastPlan, configs: Optional[list[ExperimentConfig]] = None
    ) -> pl.DataFrame:
        from joblib import Parallel, delayed

        configs = list(configs) if configs is not None else ExperimentGrid(self.settings).configs(model)
        # heavy specs first -> better load balance
        configs.sort(key=lambda c: (c.var_set != "all", c.features_type != "rolling"))
        n_jobs = {
            "dfm": min(self.n_jobs, config.RT_DFM_N_JOBS),
            "mfbvar": min(self.n_jobs, config.RT_MFBVAR_N_JOBS),
        }.get(model, self.n_jobs)
        # MF-BVAR is re-estimated at every origin, the other models once per plan segment
        n_points = sum(len(block) for *_, block in plan.segments) if model == "mfbvar" else len(plan.segments)
        logger.info(
            f"[{model}] {len(configs)} specifications, {n_points} estimation points, {n_jobs} workers"
        )

        started, frames = time.time(), []
        tasks = (delayed(run_task)(self.dataset, self.settings, cfg, plan, self.sink) for cfg in configs)
        results = Parallel(n_jobs=n_jobs, return_as="generator_unordered", batch_size=1)(tasks)
        for done, (cfg, frame, error) in enumerate(results, start=1):
            if error:
                self.errors.append({"model": model, "spec": cfg.label, "error": error})
                logger.error(f"[{model}] {cfg.label} failed: {error}")
            elif frame is not None and frame.height:
                frames.append(frame)
            if done % max(1, len(configs) // 20) == 0 or done == len(configs):
                elapsed = time.time() - started
                logger.info(
                    f"[{model}] {done}/{len(configs)} done, {elapsed / 60:.1f} min, eta {elapsed / done * (len(configs) - done) / 60:.1f} min"
                )
        result = pl.concat(frames) if frames else pl.DataFrame(schema=FORECAST_SCHEMA)
        if model == "mfbvar":  # MF-BVAR is estimated at every origin: refit_date < origin = fallback
            stale = result.filter(pl.col("refit_date") < pl.col("origin")).select(*CONFIG_COLUMNS, "origin").unique()
            if stale.height:
                logger.warning(f"[{model}] {stale.height} spec x origin forecasts taken from an earlier origin")
        return result

    def retry(self, failed: pl.DataFrame, plan: ForecastPlan, checkpoint_dir: Path) -> pl.DataFrame:
        """Re-run failed specifications and merge them into the model checkpoints."""
        frames = []
        for model, group in failed.group_by("model", maintain_order=True):
            model = model[0] if isinstance(model, tuple) else model
            configs = [ExperimentConfig.from_label(label) for label in group["spec"].to_list()]
            logger.info(f"[{model}] retrying {len(configs)} failed specifications")
            result = self.run_model(model, plan, configs)
            checkpoint = checkpoint_dir / f"forecasts_{model}.parquet"
            if checkpoint.exists():
                result = pl.concat([pl.read_parquet(checkpoint), result], how="diagonal").unique(
                    subset=[*CONFIG_COLUMNS, "target", "horizon", "origin"], keep="last"
                )
            result.write_parquet(checkpoint)
            frames.append(result)
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame(schema=FORECAST_SCHEMA)
