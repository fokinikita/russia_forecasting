"""Stages of the weekly run. Every stage pulls what it needs from S3, computes, and pushes its results,
so any stage can be re-run alone (also in a fresh container) — see service/layout.py for the S3 layout.

    collect   -> raw_data/<date>/
    backtest  -> retro_metrics/<date>/forecasts_<model>.parquet      (one model per call)
    evaluate  -> retro_metrics/<date>/ metrics, selected specs ...   + history/metrics_history
    future    -> forecasts/<date>/, models/<date>/                    + history/forecasts_history
    cbr       -> cbr_forecasts/
    dashboard -> dashboard/<date>/, dashboard/latest/
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

import attrs
import polars as pl

import config
from forecasting.experiment import ExperimentRunner, ForecastPlan
from forecasting.pipeline import RealTimePipeline
from service import layout as L
from service.cbr import SCHEMA as CBR_SCHEMA
from service.cbr import CbrForecastParser, CbrForecastScraper, dashboard_payload
from service.layout import RunLayout
from service.settings import experiment_settings
from service.storage import S3Storage

logger = logging.getLogger(__name__)


class StageError(RuntimeError):
    pass


@attrs.define
class StageContext:
    run_date: str
    storage: S3Storage
    profile: str = "full"
    n_jobs: int = config.RT_N_JOBS
    force: bool = False  # recompute even if this run date already has the result in S3
    layout: RunLayout = attrs.field(default=None)

    def __attrs_post_init__(self) -> None:
        date.fromisoformat(self.run_date)  # validates the format
        if self.layout is None:
            self.layout = RunLayout(self.run_date)

    def pipeline(self, models: tuple[str, ...] = tuple(config.RT_MODELS)) -> RealTimePipeline:
        return RealTimePipeline(
            models=models,
            settings=experiment_settings(self.profile),
            results_dir=self.layout.results_dir,
            run_id=self.run_date,
            n_jobs=self.n_jobs,
        )

    # ----------------------------------------------------------- shared pulls
    def pull_raw(self) -> None:
        """IEP data of this run date -> data/iep (the forecasting code reads it there)."""
        paths = self.storage.download_dir(self.s3(L.RAW), self.layout.data_dir)
        if not any(p.name == "quarterly.parquet" for p in paths):
            raise StageError(f"no raw data for {self.run_date} in {self.storage.uri(self.s3(L.RAW))}, run 'collect' first")

    def pull_history(self, name: str) -> None:
        self.storage.download(self.storage.key(L.HISTORY, f"{name}.parquet"), self.layout.results_dir)

    def push_history(self, name: str) -> None:
        path = self.layout.results_dir / f"{name}.csv"
        if path.exists():
            self.storage.upload(path, self.storage.key(L.HISTORY))

    def s3(self, stage: str, run_date: Optional[str] = None) -> str:
        return self.storage.key(self.layout.s3(stage, run_date))


# ------------------------------------------------------------------ collect
@attrs.define
class CollectStage:
    ctx: StageContext

    def run(self) -> dict:
        from data_iep.pipeline import IepDataPipeline

        data_dir = self.ctx.layout.data_dir
        source = "data.iep.ru"
        try:
            pipeline = IepDataPipeline(output_dir=data_dir)
            pipeline.run()
            errors = pipeline.errors
        except Exception as error:
            # site down: continue with the latest data saved in S3 (or shipped with the image)
            logger.error(f"IEP download failed: {error}")
            errors = {"all": str(error)}
            source = self._fallback()
        info = {"run_date": self.ctx.run_date, "source": source, "failed_series": errors, "tables": {}}
        for name in ("quarterly", "monthly", "daily"):
            frame = pl.read_parquet(data_dir / f"{name}.parquet")
            info["tables"][name] = {"rows": frame.height, "columns": frame.width - 1, "last_date": str(frame["date"].max())}
        (data_dir / "collect_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        files = [p for p in data_dir.iterdir() if p.is_file() and p.suffix in (".parquet", ".csv", ".json")]
        self.ctx.storage.upload_dir(data_dir, self.ctx.s3(L.RAW), files)
        logger.info(f"raw data: {json.dumps(info['tables'], ensure_ascii=False)}, {len(errors)} series failed")
        return info

    def _fallback(self) -> str:
        previous = [d for d in self.ctx.storage.folders(self.ctx.storage.key(L.RAW)) if d < self.ctx.run_date]
        if previous:
            self.ctx.storage.download_dir(self.ctx.s3(L.RAW, previous[-1]), self.ctx.layout.data_dir)
            logger.warning(f"using raw data of {previous[-1]} from S3")
            return f"fallback: raw_data/{previous[-1]}"
        if (self.ctx.layout.data_dir / "quarterly.parquet").exists():
            logger.warning("using the data shipped with the image")
            return "fallback: image copy"
        raise StageError("IEP is unavailable and there is no saved data")


# ----------------------------------------------------------------- backtest
@attrs.define
class BacktestStage:
    """Pseudo real-time backtest of one model (one Airflow task per model)."""

    ctx: StageContext
    model: str

    def run(self) -> dict:
        out_dir = self.ctx.layout.backtest_dir()
        checkpoint = out_dir / f"forecasts_{self.model}.parquet"
        folder = self.ctx.s3(L.RETRO)
        key = f"{folder}/{checkpoint.name}"
        if not self.ctx.force and self.ctx.storage.exists(key):
            logger.info(f"[{self.model}] backtest of {self.ctx.run_date} is already in S3, skipped (use --force to redo)")
            return {"model": self.model, "skipped": True}

        self.ctx.pull_raw()
        pipeline = self.ctx.pipeline((self.model,))
        dataset = pipeline.load_dataset()
        plan = ForecastPlan.backtest(dataset, pipeline.settings)
        runner = ExperimentRunner(dataset, pipeline.settings, n_jobs=self.ctx.n_jobs)
        forecasts = runner.run((self.model,), plan, checkpoint_dir=out_dir, resume=not self.ctx.force)
        failed = out_dir / f"failed_specs_{self.model}.csv"
        failed.unlink(missing_ok=True)
        if runner.errors:
            pl.DataFrame(runner.errors).write_csv(failed)
        if forecasts.height == 0:
            raise StageError(f"[{self.model}] no backtest forecasts ({len(runner.errors)} specs failed)")
        self.ctx.storage.upload_dir(out_dir, folder, [p for p in (checkpoint, failed) if p.exists()])
        logger.info(f"[{self.model}] {forecasts.height} backtest forecasts, {len(runner.errors)} failed specs")
        return {"model": self.model, "rows": forecasts.height, "failed_specs": len(runner.errors)}


# ----------------------------------------------------------------- evaluate
@attrs.define
class EvaluateStage:
    """Test metrics of every model that has backtest forecasts, automatic choice of specs."""

    ctx: StageContext

    def run(self) -> dict:
        out_dir = self.ctx.layout.backtest_dir()
        self.ctx.pull_raw()
        for old in out_dir.glob("forecasts_*.parquet"):
            old.unlink()  # only what is in S3 for this date counts
        self.ctx.storage.download_dir(
            self.ctx.s3(L.RETRO), out_dir, include=lambda n: n.startswith(("forecasts_", "failed_specs_"))
        )
        checkpoints = sorted(out_dir.glob("forecasts_*.parquet"))
        if not checkpoints:
            raise StageError(f"no backtest forecasts for {self.ctx.run_date}")
        models = tuple(p.stem.removeprefix("forecasts_") for p in checkpoints)
        missing = sorted(set(config.RT_MODELS) - set(models))
        if missing:
            logger.warning(f"no backtest for {missing}, evaluating {list(models)}")

        self.ctx.pull_history("metrics_history")
        pipeline = self.ctx.pipeline(models)
        pipeline.load_dataset()
        pipeline.save_run_info()
        forecasts = pl.concat([pl.read_parquet(p) for p in checkpoints], how="diagonal")
        summary = pipeline.evaluate(forecasts)
        failed = [pl.read_csv(p) for p in sorted(out_dir.glob("failed_specs_*.csv"))]
        if failed:
            pl.concat(failed, how="diagonal").write_csv(out_dir / "failed_specs.csv")

        results = [p for p in out_dir.iterdir() if p.is_file() and not p.name.startswith(("forecasts_", "failed_specs_"))]
        self.ctx.storage.upload_dir(out_dir, self.ctx.s3(L.RETRO), results)
        self.ctx.storage.upload(pipeline.run_dir / "run_info.json", self.ctx.s3(L.RETRO))
        self.ctx.push_history("metrics_history")
        best = pl.read_csv(out_dir / "selected_specs.csv").filter(pl.col("model_rank") == 1)
        for row in best.filter(pl.col("availability") == 1).iter_rows(named=True):
            logger.info(
                f"best a1 {row['target']}: {row['model']}/{row['train_start']}/{row['var_set']}/{row['features_type']}"
                f" rel RMSE {row['mean_rel_rmse']:.3f}"
            )
        return {"models": list(models), "specs": summary.height}


# ------------------------------------------------------------------- future
@attrs.define
class FutureStage:
    """2-year-ahead forecasts from the latest data, fitted models of the chosen specs are kept."""

    ctx: StageContext
    models: tuple[str, ...] = tuple(config.RT_MODELS)

    def run(self) -> dict:
        ctx, lay = self.ctx, self.ctx.layout
        ctx.pull_raw()
        retro = ctx.storage.download_dir(
            ctx.s3(L.RETRO), lay.backtest_dir(), include=lambda n: not n.startswith(("forecasts_", "failed_specs"))
        )
        if not any(p.name == "selected_specs.csv" for p in retro):
            raise StageError(f"no backtest metrics for {ctx.run_date}, run 'evaluate' first")
        shutil.copy(lay.backtest_dir() / "run_info.json", lay.run_dir() / "run_info.json")

        future_dir = lay.future_dir()
        if ctx.force:
            shutil.rmtree(future_dir, ignore_errors=True)
            shutil.rmtree(lay.models_dir(), ignore_errors=True)
        else:  # resume: models already forecast for this date are not recomputed
            ctx.storage.download_dir(ctx.s3(L.FORECASTS), future_dir, include=lambda n: n.startswith("forecasts_"))
        previous = self._pull_previous_run()
        ctx.pull_history("forecasts_history")

        pipeline = ctx.pipeline(self.models)
        forecasts = pipeline.forecast_future(resume=not ctx.force, save_models=True)
        ctx.storage.upload_dir(future_dir, ctx.s3(L.FORECASTS))
        n_models = ctx.storage.upload_dir(lay.models_dir(), ctx.s3(L.MODELS))
        ctx.push_history("forecasts_history")
        logger.info(f"future: {forecasts.height} forecasts, {len(n_models)} model files, fallback run {previous}")
        return {"rows": forecasts.height, "model_files": len(n_models), "previous_run": previous}

    def _pull_previous_run(self) -> Optional[str]:
        """Future forecasts of the latest earlier run: failed specs of this run are filled from it."""
        runs = [d for d in self.ctx.storage.folders(self.ctx.storage.key(L.FORECASTS)) if d < self.ctx.run_date]
        if not runs:
            return None
        self.ctx.storage.download_dir(
            self.ctx.s3(L.FORECASTS, runs[-1]),
            self.ctx.layout.future_dir(runs[-1]),
            include=lambda n: n.startswith("forecasts_"),
        )
        return runs[-1]


# ---------------------------------------------------------------------- cbr
@attrs.define
class CbrStage:
    """Checks cbr.ru for new medium-term forecasts, parses new PDFs, keeps all of them in S3."""

    ctx: StageContext
    scraper: CbrForecastScraper = attrs.field(factory=CbrForecastScraper)
    parser: CbrForecastParser = attrs.field(factory=CbrForecastParser)
    since_year: int = 2022  # older PDFs have other layouts and are not needed for the comparison

    @property
    def all_key(self) -> str:
        return self.ctx.storage.key(L.CBR, "all_forecasts.parquet")

    def load_all(self) -> pl.DataFrame:
        data = self.ctx.storage.get_bytes(self.all_key)
        return pl.read_parquet(data) if data else pl.DataFrame(schema=CBR_SCHEMA)

    def run(self) -> dict:
        known = self.load_all()
        known_files = set(known["file"].unique().to_list())
        refs = [r for r in self.scraper.list_forecasts() if r.forecast_date.year >= self.since_year]
        new = [r for r in refs if r.file not in known_files]
        frames, failed = [known], []
        for ref in sorted(new):
            pdf_key = self.ctx.storage.key(L.CBR, "pdf", ref.file)
            data = self.ctx.storage.get_bytes(pdf_key)
            if data is None:
                data = self.scraper.download(ref)
                self.ctx.storage.put_bytes(data, pdf_key, "application/pdf")
            frame = self.parser.parse_pdf(data, ref)
            if frame.height == 0:
                failed.append(ref.file)
                logger.warning(f"{ref.file}: forecast table not found")
                continue
            logger.info(f"new CBR forecast {ref.forecast_date}: {frame.height} values")
            frames.append(frame)
        forecasts = pl.concat(frames, how="diagonal_relaxed").sort(["forecast_date", "target", "year"])
        if new:
            self.ctx.storage.put_bytes(_parquet_bytes(forecasts), self.all_key, "application/vnd.apache.parquet")
        snapshot = forecasts.filter(pl.col("forecast_date") <= date.fromisoformat(self.ctx.run_date))
        self.ctx.storage.put_bytes(
            _parquet_bytes(snapshot), f"{self.ctx.s3(L.CBR)}/cbr_forecasts.parquet", "application/vnd.apache.parquet"
        )
        latest = [p["date"] for p in dashboard_payload(snapshot)]
        logger.info(f"CBR: {len(new) - len(failed)} new forecasts, latest three: {latest}")
        return {"new": [r.file for r in new if r.file not in failed], "failed": failed, "latest": latest}


def _parquet_bytes(frame: pl.DataFrame) -> bytes:
    import io

    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------- dashboard
@attrs.define
class DashboardStage:
    ctx: StageContext

    def run(self) -> dict:
        from forecasting.dashboard import build_dashboard

        ctx, lay = self.ctx, self.ctx.layout
        ctx.pull_raw()
        ctx.storage.download_dir(
            ctx.s3(L.RETRO), lay.backtest_dir(), include=lambda n: not n.startswith(("forecasts_", "failed_specs_"))
        )
        future = ctx.storage.download_dir(ctx.s3(L.FORECASTS), lay.future_dir(), include=lambda n: not n.startswith("forecasts_"))
        if not any(p.name == "forecasts.csv" for p in future):
            raise StageError(f"no future forecasts for {ctx.run_date}, run 'future' first")
        shutil.copy(lay.backtest_dir() / "run_info.json", lay.run_dir() / "run_info.json")

        cbr = self._cbr_payload()
        path = build_dashboard(
            run_id=ctx.run_date,
            as_of=date.fromisoformat(ctx.run_date),
            out_path=lay.dashboard_path(),
            results_dir=str(lay.results_dir),
            cbr=cbr,
        )
        key = ctx.storage.upload(path, ctx.s3(L.DASHBOARD))
        latest = ctx.storage.upload(path, ctx.storage.key(L.DASHBOARD, "latest"))
        url = ctx.storage.presigned_url(key)
        logger.info(f"dashboard: {ctx.storage.uri(key)}\nlink for 7 days: {url}")
        return {"key": key, "latest": latest, "url": url, "cbr_forecasts": [c["date"] for c in cbr]}

    def _cbr_payload(self) -> list:
        storage = self.ctx.storage
        data = storage.get_bytes(f"{self.ctx.s3(L.CBR)}/cbr_forecasts.parquet") or storage.get_bytes(
            storage.key(L.CBR, "all_forecasts.parquet")
        )
        if not data:
            logger.warning("no Bank of Russia forecasts in S3, the dashboard is built without them")
            return []
        frame = pl.read_parquet(data).filter(pl.col("forecast_date") <= date.fromisoformat(self.ctx.run_date))
        return dashboard_payload(frame)


# --------------------------------------------------------------------- seed
@attrs.define
class SeedStage:
    """One-off: upload a run computed before the service existed (results/runs/<date>) to S3."""

    ctx: StageContext
    source_results: Path
    source_data: Path

    def run(self) -> dict:
        ctx, run_dir = self.ctx, Path(self.source_results) / "runs" / self.ctx.run_date
        if not run_dir.exists():
            raise StageError(f"{run_dir} does not exist")
        files = [p for p in Path(self.source_data).iterdir() if p.is_file() and p.suffix in (".parquet", ".csv")]
        ctx.storage.upload_dir(self.source_data, ctx.s3(L.RAW), files)
        ctx.storage.upload_dir(run_dir / "backtest", ctx.s3(L.RETRO))
        ctx.storage.upload(run_dir / "run_info.json", ctx.s3(L.RETRO))
        ctx.storage.upload_dir(run_dir / "future", ctx.s3(L.FORECASTS))
        for name in ("metrics_history", "forecasts_history"):
            path = Path(self.source_results) / f"{name}.csv"
            if path.exists() and not ctx.storage.exists(ctx.storage.key(L.HISTORY, f"{name}.parquet")):
                ctx.storage.upload(path, ctx.storage.key(L.HISTORY))
        return {"seeded": ctx.run_date}
