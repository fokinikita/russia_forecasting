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
    runs/<run_id>/backtest/rolling.csv                 rolling RMSE per test quarter (quality dynamics)
    runs/<run_id>/future/forecasts.csv                 all specs, 8 quarters ahead
    runs/<run_id>/future/best.csv                      forecast of the best backtest spec
    runs/<run_id>/future/consensus.csv                 median of the top-K backtest specs (recommended)
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
from forecasting.experiment import CONFIG_COLUMNS, ExperimentRunner, ExperimentSettings, ForecastPlan

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
        forecasts = forecasts.with_columns(((pl.col("pred").exp() - 1) * 100).alias("pred_yoy_pct")).sort(
            [*SPEC_COLUMNS]
        )
        forecasts.write_csv(out_dir / "forecasts.csv")

        best = self._best_future(forecasts)
        consensus = self._consensus_future(forecasts)
        if best is not None and consensus is not None:
            best.write_csv(out_dir / "best.csv")
            consensus.write_csv(out_dir / "consensus.csv")
            keys = ["target", "availability", "horizon", "origin", "target_date"]
            history = pl.concat(
                [
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
