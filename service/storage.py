"""S3 storage of the service.

Tables are stored as parquet: CSV files written by the forecasting code are converted to parquet on
upload and back to CSV on download (the code reads them as CSV), everything else (parquet, json, html,
joblib, pdf) is stored as is.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import attrs
import polars as pl

from service.settings import S3Settings

logger = logging.getLogger(__name__)

# stems of the tables the forecasting code reads / writes as CSV
CSV_STEMS = frozenset(
    {
        "metadata",
        "metrics",
        "ranking",
        "best",
        "selected_specs",
        "rolling",
        "failed_specs",
        "forecasts",
        "consensus",
        "selected",
        "metrics_history",
        "forecasts_history",
    }
)


def is_csv_table(stem: str) -> bool:
    return stem in CSV_STEMS or stem.startswith("failed_specs")


def csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    # dates stay text: run ids like "2026-10-22" must come back unchanged
    frame = pl.read_csv(csv_path, infer_schema_length=None)
    frame.write_parquet(parquet_path)


def parquet_to_csv(parquet_path: Path, csv_path: Path) -> None:
    pl.read_parquet(parquet_path).write_csv(csv_path)


@attrs.define
class S3Storage:
    settings: S3Settings
    _client: object = attrs.field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls, prefix: Optional[str] = None) -> "S3Storage":
        return cls(S3Settings.from_env(prefix))

    # ------------------------------------------------------------ client
    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.endpoint_url,
                aws_access_key_id=self.settings.access_key,
                aws_secret_access_key=self.settings.secret_key,
                region_name=self.settings.region,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    # S3-compatible storages reject the new default CRC checksums of boto3 >= 1.36
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                    retries={"max_attempts": 8, "mode": "standard"},
                ),
            )
        return self._client

    @property
    def bucket(self) -> str:
        return self.settings.bucket

    def key(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in (self.settings.prefix, *parts) if p and p.strip("/"))

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    # ------------------------------------------------------------ objects
    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str) -> list[str]:
        """All keys under `prefix` (a folder: a trailing "/" is added)."""
        prefix = prefix.rstrip("/") + "/" if prefix.strip("/") else ""
        keys, token = [], None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            keys += [obj["Key"] for obj in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                return keys
            token = page["NextContinuationToken"]

    def folders(self, prefix: str) -> list[str]:
        """Names of the direct sub-folders of `prefix` (e.g. the run dates)."""
        prefix = prefix.rstrip("/") + "/" if prefix.strip("/") else ""
        names, token = [], None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix, "Delimiter": "/"}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            names += [p["Prefix"][len(prefix) :].strip("/") for p in page.get("CommonPrefixes", [])]
            if not page.get("IsTruncated"):
                return sorted(names)
            token = page["NextContinuationToken"]

    def upload_file(self, path: Path, key: str) -> str:
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".json": "application/json",
            ".pdf": "application/pdf",
            ".parquet": "application/vnd.apache.parquet",
        }.get(Path(path).suffix, "application/octet-stream")
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": content_type})
        logger.info(f"uploaded {path} -> {self.uri(key)}")
        return key

    def download_file(self, key: str, path: Path) -> bool:
        """Download `key` to `path`; False if there is no such object."""
        if not self.exists(key):
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(path))
        return True

    def put_bytes(self, data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        logger.info(f"uploaded {len(data) / 1e3:.0f} kB -> {self.uri(key)}")
        return key

    def get_bytes(self, key: str) -> Optional[bytes]:
        if not self.exists(key):
            return None
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def presigned_url(self, key: str, expires: int = 7 * 24 * 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires
        )

    # ------------------------------------------------------------- tables
    def upload(self, path: Path, folder: str) -> str:
        """Upload one result file into `folder`; CSV tables become parquet."""
        path = Path(path)
        if path.suffix == ".csv":
            with tempfile.TemporaryDirectory() as tmp:
                parquet = Path(tmp) / f"{path.stem}.parquet"
                csv_to_parquet(path, parquet)
                return self.upload_file(parquet, f"{folder}/{path.stem}.parquet")
        return self.upload_file(path, f"{folder}/{path.name}")

    def upload_dir(self, local_dir: Path, folder: str, files: Optional[Iterable[Path]] = None) -> list[str]:
        """Upload every file of `local_dir` (recursively, or only `files`) keeping sub-folders."""
        local_dir = Path(local_dir)
        if not local_dir.exists():
            return []
        paths = sorted(files) if files is not None else sorted(p for p in local_dir.rglob("*") if p.is_file())
        keys = []
        for path in paths:
            relative = path.relative_to(local_dir).parent.as_posix()
            sub = folder if relative == "." else f"{folder}/{relative}"
            keys.append(self.upload(path, sub))
        return keys

    def download(self, key: str, local_dir: Path) -> Optional[Path]:
        """Download one stored file into `local_dir`; parquet copies of CSV tables come back as CSV."""
        local_dir = Path(local_dir)
        name = key.rsplit("/", 1)[-1]
        stem, _, suffix = name.rpartition(".")
        if suffix == "parquet" and is_csv_table(stem):
            with tempfile.TemporaryDirectory() as tmp:
                parquet = Path(tmp) / name
                if not self.download_file(key, parquet):
                    return None
                target = local_dir / f"{stem}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                parquet_to_csv(parquet, target)
                return target
        target = local_dir / name
        return target if self.download_file(key, target) else None

    def download_dir(self, folder: str, local_dir: Path, include=None) -> list[Path]:
        """Download everything under `folder` (optionally only names for which include(name) is true)."""
        local_dir = Path(local_dir)
        paths = []
        for key in self.list(folder):
            relative = key[len(folder.rstrip("/")) + 1 :]
            name = relative.rsplit("/", 1)[-1]
            if include is not None and not include(name):
                continue
            sub = relative.rsplit("/", 1)[0] if "/" in relative else ""
            path = self.download(key, local_dir / sub if sub else local_dir)
            if path is not None:
                paths.append(path)
        logger.info(f"downloaded {len(paths)} files {self.uri(folder)} -> {local_dir}")
        return paths
