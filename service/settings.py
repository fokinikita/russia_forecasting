"""Service settings from environment variables (see .env.example)."""

from __future__ import annotations

import os

import attrs

from forecasting.experiment import ExperimentSettings

import config


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise RuntimeError(f"environment variable {name} is not set (see .env.example)")
    return value


@attrs.frozen
class S3Settings:
    endpoint_url: str
    bucket: str
    access_key: str = attrs.field(repr=False)
    secret_key: str = attrs.field(repr=False)
    region: str = "ru-1"
    prefix: str = ""  # optional root folder inside the bucket, e.g. "_smoke" for test runs

    @classmethod
    def from_env(cls, prefix: str | None = None) -> "S3Settings":
        return cls(
            endpoint_url=_env("S3_ENDPOINT_URL", "https://s3.twcstorage.ru"),
            bucket=_env("S3_BUCKET"),
            access_key=_env("S3_ACCESS_KEY"),
            secret_key=_env("S3_SECRET_KEY"),
            region=_env("S3_REGION", "ru-1"),
            prefix=(prefix if prefix is not None else os.environ.get("S3_PREFIX", "")).strip("/"),
        )


# Experiment grids the service can run. "full" = the research grid of config.py,
# "quick" = a small grid to check the whole service end to end in ~15 minutes.
PROFILES = ("full", "quick")


def experiment_settings(profile: str = "full") -> ExperimentSettings:
    if profile == "full":
        return ExperimentSettings()
    if profile == "quick":
        sets = config.RT_VARIABLE_SETS
        return ExperimentSettings(
            train_starts=(2010,),
            feature_types=("classic",),
            variable_sets={"short": sets["short"], "medium": sets["medium"]},
        )
    raise ValueError(f"unknown profile {profile!r}, expected one of {PROFILES}")
