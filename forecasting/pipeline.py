"""Real-time forecasting pipeline.

    refresh IEP data -> quarterly dataset -> backtest on the last 3 years
    -> test metrics -> 2-year-ahead forecasts from the latest data

Layout of results (config.RT_RESULTS_DIR):

    runs/<run_id>/run_info.json
    runs/<run_id>/backtest/forecasts_<model>.parquet   checkpoints (a rerun resumes from them)
    runs/<run_id>/backtest/errors.parquet              every test forecast with fact and naive
    runs/<run_id>/backtest/metrics.csv                 RMSE/MAE/bias per spec, target, horizon
    runs/<run_id>/backtest/ranking.csv                 specs ranked by mean RMSE relative to naive
    runs/<run_id>/backtest/best.csv                    best spec per target x horizon x availability
    runs/<run_id>/backtest/selected_specs.csv          auto-chosen train start / var set / features type
                                                       per model x target x availability, models ranked
    runs/<run_id>/backtest/rolling.csv                 rolling RMSE per test quarter (quality dynamics)
    runs/<run_id>/future/forecasts.csv                 all specs, 8 quarters ahead
    runs/<run_id>/future/best.csv                      forecast of the best backtest spec
    runs/<run_id>/future/consensus.csv                 median of the top-K backtest specs (recommended)
    runs/<run_id>/future/selected.csv                  forecasts of the auto-chosen spec of every model
                                                       (model_rank = 1 -> "auto" forecast)
    metrics_history.csv                                metrics of every run (quality across runs)
    forecasts_history.csv                              consensus and best-spec forecasts of every run
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import attrs
import polars as pl

import config
from forecasting.dataset import DatasetBuilder, QuarterlyDataset
from forecasting.evaluation import SPEC_COLUMNS, ForecastEvaluator
from forecasting.experiment import (
    CONFIG_COLUMNS,
    SYSTEM_MODELS,
    ExperimentGrid,
    ExperimentRunner,
    ExperimentSettings,
    ForecastPlan,
)

logger = logging.getLogger(__name__)


def _append_csv(path: Path, frame: pl.DataFrame) -> None:
    # dates are stored as text: run ids like "2026-10-22" must never be parsed as dates
    frame = frame.with_columns(pl.col(pl.Date).cast(pl.Utf8))
    if path.exists():
        previous = pl.read_csv(
            path,
            schema_overrides={
                "run_id": pl.Utf8,
                "last_known_quarter": pl.Utf8,
                "origin": pl.Utf8,
                "target_date": pl.Utf8,
            },
        )
        run_ids = frame["run_id"].unique().to_list()
        previous = previous.filter(~pl.col("run_id").is_in(run_ids))  # a rerun replaces its own rows
        frame = pl.concat([previous, frame], how="diagonal_relaxed")
    frame.write_csv(path)


@attrs.define(slots=True)
class RealTimePipeline:
    models: tuple[str, ...] = tuple(config.RT_MODELS)
    settings: ExperimentSettings = attrs.field(factory=ExperimentSettings)
    results_dir: Path = attrs.field(default=Path(config.RT_RESULTS_DIR), converter=Path)
    run_id: str = attrs.field(factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    n_jobs: int = config.RT_N_JOBS
    dataset: Optional[QuarterlyDataset] = None

    @property
    def run_dir(self) -> Path:
        return self.results_dir / "runs" / self.run_id

    # -------------------------------------------------------------- data
    def refresh_data(self) -> None:
        """Download the latest IEP data; on failure keep working with the saved copy."""
        from data_iep.pipeline import IepDataPipeline

        try:
            pipeline = IepDataPipeline()
            pipeline.run()
            if pipeline.errors:
                logger.warning(
                    f"IEP: {len(pipeline.errors)} series failed, previous values kept: {list(pipeline.errors)}"
                )
        except Exception as error:
            logger.error(f"IEP refresh failed ({error}), using the data saved in {config.IEP_DATA_DIR}")

    def load_dataset(self) -> QuarterlyDataset:
        self.dataset = DatasetBuilder(targets=self.settings.targets).build_from_iep()
        return self.dataset

    # ---------------------------------------------------------- backtest
    def backtest(self, resume: bool = True) -> pl.DataFrame:
        dataset = self.dataset or self.load_dataset()
        plan = ForecastPlan.backtest(dataset, self.settings)
        logger.info(
            f"backtest: test {min(plan.target_dates)} -> {max(plan.target_dates)}, "
            f"origins {plan.segments[0][0]} -> {plan.segments[-1][2][-1]}, {len(plan.segments)} estimation points"
        )
        out_dir = self.run_dir / "backtest"
        runner = ExperimentRunner(dataset, self.settings, n_jobs=self.n_jobs)
        forecasts = runner.run(self.models, plan, checkpoint_dir=out_dir, resume=resume)
        self._save_errors(runner.errors, out_dir / "failed_specs.csv")
        self.evaluate(forecasts)
        return forecasts

    def evaluate(self, forecasts: pl.DataFrame) -> pl.DataFrame:
        out_dir = self.run_dir / "backtest"
        out_dir.mkdir(parents=True, exist_ok=True)
        evaluator = ForecastEvaluator(forecasts, self.dataset or self.load_dataset())
        errors = evaluator.errors()
        summary = evaluator.summary(errors)

        errors.write_parquet(out_dir / "errors.parquet")
        summary.write_csv(out_dir / "metrics.csv")
        evaluator.ranking(summary).write_csv(out_dir / "ranking.csv")
        evaluator.best(summary).write_csv(out_dir / "best.csv")
        evaluator.selected(summary, config.RT_SELECTION_METRIC).write_csv(out_dir / "selected_specs.csv")
        evaluator.rolling(errors).write_csv(out_dir / "rolling.csv")
        _append_csv(
            self.results_dir / "metrics_history.csv",
            summary.with_columns(
                pl.lit(self.run_id).alias("run_id"),
                pl.lit(str(self.dataset.last_target_date)).alias("last_known_quarter"),
            ),
        )
        logger.info(f"metrics saved to {out_dir}")
        return summary

    # ------------------------------------------------------------ future
    def forecast_future(self, resume: bool = True) -> pl.DataFrame:
        dataset = self.dataset or self.load_dataset()
        plan = ForecastPlan.future(dataset)
        out_dir = self.run_dir / "future"
        logger.info(f"future: origin {plan.segments[0][0]}, targets known up to {plan.segments[0][1]}")
        runner = ExperimentRunner(dataset, self.settings, n_jobs=self.n_jobs)
        forecasts = runner.run(self.models, plan, checkpoint_dir=out_dir, resume=resume)
        self._save_errors(runner.errors, out_dir / "failed_specs.csv")
        return self._save_future(forecasts, out_dir)

    def _save_future(self, forecasts: pl.DataFrame, out_dir: Path) -> pl.DataFrame:
        if config.RT_FUTURE_FALLBACK_TO_PREVIOUS_RUN:
            forecasts = self._fill_from_previous_runs(forecasts)
        forecasts = forecasts.with_columns(((pl.col("pred").exp() - 1) * 100).alias("pred_yoy_pct")).sort(
            [*SPEC_COLUMNS]
        )
        forecasts.write_csv(out_dir / "forecasts.csv")

        best = self._best_future(forecasts)
        consensus = self._consensus_future(forecasts)
        selected = self._selected_future(forecasts)
        if best is not None and consensus is not None:
            best.write_csv(out_dir / "best.csv")
            consensus.write_csv(out_dir / "consensus.csv")
            keys = ["target", "availability", "horizon", "origin", "target_date"]
            spec = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in CONFIG_COLUMNS], separator="/").alias("spec")
            auto_parts = []
            if selected is not None:
                selected.write_csv(out_dir / "selected.csv")
                auto_parts = [
                    selected.filter(pl.col("model_rank") == 1).select(
                        *keys, pl.lit("auto").alias("method"), "pred_yoy_pct", spec
                    ),
                    selected.select(
                        *keys, pl.concat_str([pl.lit("auto_"), pl.col("model")]).alias("method"), "pred_yoy_pct", spec
                    ),
                ]
            history = pl.concat(
                [
                    *auto_parts,
                    consensus.select(
                        *keys, pl.lit("consensus").alias("method"), "pred_yoy_pct", pl.col("specs").alias("spec")
                    ),
                    best.select(
                        *keys,
                        pl.lit("best_spec").alias("method"),
                        "pred_yoy_pct",
                        pl.concat_str([pl.col(c).cast(pl.Utf8) for c in CONFIG_COLUMNS], separator="/").alias("spec"),
                    ),
                ],
                how="diagonal_relaxed",
            )
            _append_csv(
                self.results_dir / "forecasts_history.csv", history.with_columns(pl.lit(self.run_id).alias("run_id"))
            )
        logger.info(f"future forecasts saved to {out_dir}")
        return forecasts

    def _consensus_future(
        self, forecasts: pl.DataFrame, top_k: int = config.RT_CONSENSUS_TOP_K
    ) -> Optional[pl.DataFrame]:
        """Median forecast of the top-K specs (by backtest RMSE) for every target x availability x horizon.

        More robust than the single best spec: with 12 test quarters the winner is often only
        marginally better than the runner-up and can produce an outlier forecast.
        """
        metrics_path = self.run_dir / "backtest" / "metrics.csv"
        if not metrics_path.exists():
            return None
        metrics = pl.read_csv(metrics_path).select(
            *SPEC_COLUMNS, pl.col("rmse").alias("backtest_rmse"), pl.col("rel_rmse_naive")
        )
        ranked = (
            forecasts.join(metrics, on=SPEC_COLUMNS, how="inner")
            .sort("backtest_rmse")
            .with_columns(
                pl.col("backtest_rmse").rank("ordinal").over(["target", "availability", "horizon"]).alias("_rank")
            )
            .filter(pl.col("_rank") <= top_k)
        )
        return (
            ranked.group_by(["target", "availability", "horizon", "origin", "target_date"])
            .agg(
                pl.col("pred").median().alias("pred"),
                pl.col("pred_yoy_pct").median().alias("pred_yoy_pct"),
                pl.col("pred_yoy_pct").min().alias("min_yoy_pct"),
                pl.col("pred_yoy_pct").max().alias("max_yoy_pct"),
                pl.col("backtest_rmse").mean().alias("mean_backtest_rmse"),
                pl.col("rel_rmse_naive").mean().alias("mean_backtest_rel_rmse_naive"),
                pl.len().alias("n_specs"),
                pl.col("model").value_counts(sort=True).struct.field("model").str.join(",").alias("models"),
                pl.concat_str([pl.col(c).cast(pl.Utf8) for c in CONFIG_COLUMNS], separator="/")
                .str.join(";")
                .alias("specs"),
            )
            .sort(["target", "availability", "horizon"])
        )

    def _selected_future(self, forecasts: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Future path of the auto-chosen spec of every model (see ForecastEvaluator.selected)."""
        path = self.run_dir / "backtest" / "selected_specs.csv"
        if not path.exists():
            return None
        selected = pl.read_csv(path).select(
            *CONFIG_COLUMNS,
            "target",
            "model_rank",
            pl.col("mean_rel_rmse").alias("backtest_mean_rel_rmse"),
            pl.col("mean_rmse").alias("backtest_mean_rmse"),
        )
        return forecasts.join(selected, on=[*CONFIG_COLUMNS, "target"], how="inner").sort(
            ["target", "availability", "model_rank", "horizon"]
        )

    # ---------------------------------------------------- previous runs
    def _expected_specs(self) -> pl.DataFrame:
        """Every spec x target x horizon that a complete future run produces."""
        rows = []
        grid = ExperimentGrid(self.settings)
        horizons = range(1, self.settings.horizon + 1)
        for model in self.models:
            for cfg in grid.configs(model):
                if model in SYSTEM_MODELS:
                    pairs = [(t, h) for t in self.settings.targets for h in horizons]
                else:
                    pairs = [(cfg.target, cfg.horizon)]
                rows += [{**cfg.tags(), "target": t, "horizon": h} for t, h in pairs]
        return pl.DataFrame(rows)

    def _fill_from_previous_runs(self, forecasts: pl.DataFrame) -> pl.DataFrame:
        """Future forecasts that failed in this run are taken from the latest earlier run with the same spec.

        The old path is re-indexed to the current origin (horizon = quarters from the current origin + 1),
        quarters before the current origin are dropped; `source_run` shows where a row came from.
        """
        dataset = self.dataset or self.load_dataset()
        origin = ForecastPlan.future(dataset).segments[0][0]
        forecasts = forecasts.with_columns(pl.lit(None, dtype=pl.Utf8).alias("source_run"))
        spec_keys = [*CONFIG_COLUMNS, "target", "horizon"]
        missing = self._expected_specs().join(forecasts.select(spec_keys), on=spec_keys, how="anti")
        runs_dir = self.results_dir / "runs"
        if missing.height == 0 or not runs_dir.exists():
            return forecasts

        months = (pl.col("target_date").dt.year() - origin.year) * 12 + pl.col("target_date").dt.month() - origin.month
        previous = sorted((p for p in runs_dir.iterdir() if p.is_dir() and p.name < self.run_id), reverse=True)
        filled = []
        for run_dir in previous:
            paths = [run_dir / "future" / f"forecasts_{m}.parquet" for m in missing["model"].unique().to_list()]
            paths = [p for p in paths if p.exists()]
            if not paths:
                continue
            old = (
                pl.concat([pl.read_parquet(p) for p in paths], how="diagonal")
                .filter(pl.col("target_date") >= origin)
                .with_columns((months // 3 + 1).cast(pl.Int64).alias("horizon"), pl.lit(origin).alias("origin"))
                .join(missing, on=spec_keys, how="semi")
                .with_columns(pl.lit(run_dir.name).alias("source_run"))
            )
            if old.height:
                filled.append(old)
                missing = missing.join(old.select(spec_keys), on=spec_keys, how="anti")
            if missing.height == 0:
                break
        if filled:
            logger.warning(
                f"future: {sum(f.height for f in filled)} failed forecasts taken from previous runs, "
                f"{missing.height} still missing"
            )
            forecasts = pl.concat([forecasts, *filled], how="diagonal_relaxed")
        elif missing.height:
            logger.warning(f"future: {missing.height} spec x target x horizon forecasts missing, no previous run has them")
        return forecasts

    def _best_future(self, forecasts: pl.DataFrame) -> Optional[pl.DataFrame]:
        best_path = self.run_dir / "backtest" / "best.csv"
        if not best_path.exists():
            logger.warning("no backtest metrics for this run, best future forecasts are not selected")
            return None
        best = pl.read_csv(best_path).select(
            [
                *SPEC_COLUMNS,
                pl.col("rmse").alias("backtest_rmse"),
                pl.col("rel_rmse_naive").alias("backtest_rel_rmse_naive"),
            ]
        )
        return forecasts.join(best, on=SPEC_COLUMNS, how="inner").sort(["target", "availability", "horizon"])

    # ------------------------------------------------------------- retry
    def retry_failed(self) -> None:
        """Re-run specifications listed in failed_specs.csv (e.g. after a MemoryError) and re-evaluate."""
        dataset = self.dataset or self.load_dataset()
        stages = {
            "backtest": ForecastPlan.backtest(dataset, self.settings),
            "future": ForecastPlan.future(dataset),
        }
        for stage, plan in stages.items():
            out_dir = self.run_dir / stage
            failed_path = out_dir / "failed_specs.csv"
            if not failed_path.exists():
                continue
            failed = pl.read_csv(failed_path)
            runner = ExperimentRunner(dataset, self.settings, n_jobs=self.n_jobs)
            runner.retry(failed, plan, out_dir)
            failed_path.unlink()
            self._save_errors(runner.errors, failed_path)
            logger.info(f"{stage}: retried {failed.height} specs, {len(runner.errors)} still failing")
            forecasts = pl.concat(
                [pl.read_parquet(p) for p in sorted(out_dir.glob("forecasts_*.parquet"))], how="diagonal"
            )
            if stage == "backtest":
                self.evaluate(forecasts)
            else:
                self._save_future(forecasts, out_dir)

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _save_errors(errors: list, path: Path) -> None:
        if errors:
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(errors).write_csv(path)
            logger.warning(f"{len(errors)} specifications failed, see {path}")

    def save_run_info(self) -> None:
        dataset = self.dataset or self.load_dataset()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "run_id": self.run_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "last_known_quarter": str(dataset.last_target_date),
            "last_data_quarter": str(dataset.frame["date"].max()),
            "models": list(self.models),
            "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in attrs.asdict(self.settings).items()},
        }
        metadata = Path(config.IEP_DATA_DIR) / "metadata.csv"
        if metadata.exists():
            info["iep_vintages"] = pl.read_csv(metadata).select("name", "vintage_id", "vintage_date").to_dicts()
        (self.run_dir / "run_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )

    def run(self, refresh: bool = True, backtest: bool = True, future: bool = True, resume: bool = True) -> None:
        if refresh:
            self.refresh_data()
        self.load_dataset()
        self.save_run_info()
        if backtest:
            self.backtest(resume=resume)
        if future:
            self.forecast_future(resume=resume)
