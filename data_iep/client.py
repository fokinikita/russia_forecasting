"""Thin HTTP client for the data.iep.ru analytic API.

How a series is fetched:
    1. dataset uuid (from the site link .../dataset/<uuid>/chart)
       -> GET /search/datasets/family/<uuid>          -> latest vintage_id
          (fallback: GET /search/datasets?q=<name>, filtered by uuid)
    2. GET /vintage/lines_list?vintage_id=<id>        -> every series ("slice")
       of the dataset with a ready-made `selector`
    3. POST /vintage/line  json=<selector>            -> observations
"""

from __future__ import annotations

import logging
import time
from typing import Any

import attrs
import polars as pl
import requests

logger = logging.getLogger(__name__)

API_URL = "https://data.iep.ru/analytic/api/v1"

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    # the API answers 400 without a language
    "accept-language": "ru",
    "content-type": "application/json",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    ),
}


class IepApiError(RuntimeError):
    pass


@attrs.define(slots=True)
class IepClient:
    base_url: str = API_URL
    timeout: int = 60
    retries: int = 3
    backoff: float = 2.0
    session: requests.Session = attrs.field(factory=requests.Session)
    # caches live for one client instance (= one pipeline run)
    _vintage_cache: dict = attrs.field(factory=dict)
    _slices_cache: dict = attrs.field(factory=dict)

    def __attrs_post_init__(self) -> None:
        self.session.headers.update(DEFAULT_HEADERS)

    # ------------------------------------------------------------------ http
    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as error:
                problem = str(error)
            else:
                if response.status_code < 400:
                    return response.json()
                problem = f"{response.status_code}: {response.text[:300]}"
                if response.status_code < 500:
                    # client errors will not fix themselves on retry
                    raise IepApiError(f"{method} {path} -> {problem}")
            logger.warning(f"{method} {path} failed ({attempt}/{self.retries}): {problem}")
            if attempt < self.retries:
                time.sleep(self.backoff * attempt)
        raise IepApiError(f"{method} {path} failed after {self.retries} attempts: {problem}")

    # --------------------------------------------------------------- vintage
    def get_latest_vintage(self, dataset_uuid: str, search_query: str | None = None) -> dict:
        """Metadata of the newest vintage (release) of a dataset."""
        if dataset_uuid in self._vintage_cache:
            return self._vintage_cache[dataset_uuid]

        family = self._request("GET", f"/search/datasets/family/{dataset_uuid}")
        items = family.get("items") or []

        # some datasets are not indexed by the family endpoint -> full-text search
        if not items and search_query:
            found = self._request(
                "GET",
                "/search/datasets",
                params={"q": search_query, "page_size": 50, "page_number": 1},
            )
            items = [item for item in found.get("items") or [] if item.get("dataset_uuid") == dataset_uuid]

        if not items:
            raise IepApiError(f"No vintage found for dataset {dataset_uuid} (search_query={search_query!r})")

        latest = max(items, key=lambda item: (item.get("vintage_date") or "", item["vintage_id"]))
        self._vintage_cache[dataset_uuid] = latest
        return latest

    def get_slices(self, vintage_id: int) -> list[dict]:
        """All series of a vintage: [{dimension, frequency, unit, selector, ...}]."""
        if vintage_id not in self._slices_cache:
            blocks = self._request("GET", "/vintage/lines_list", params={"vintage_id": vintage_id})
            self._slices_cache[vintage_id] = [line for block in blocks for line in block.get("slices", [])]
        return self._slices_cache[vintage_id]

    # ------------------------------------------------------------------ data
    def get_line(self, selector: dict) -> dict:
        return self._request("POST", "/vintage/line", json=selector)

    def get_line_frame(self, selector: dict) -> pl.DataFrame:
        """Observations as a DataFrame[date: Date, value: Float64] (date = period end as given)."""
        payload = self.get_line(selector)
        observations = payload.get("observations") or []
        if not observations:
            return pl.DataFrame(schema={"date": pl.Date, "value": pl.Float64})
        return (
            pl.DataFrame(observations)
            .select(
                pl.col("date").str.to_date("%Y-%m-%d"),
                pl.col("obs").cast(pl.Float64, strict=False).alias("value"),
            )
            .drop_nulls()
            .sort("date")
        )
