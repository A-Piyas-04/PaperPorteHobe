"""Data layer: the system of record for harvested papers and citation history.

* Raw layer: Parquet files partitioned by month of first submission,
  ``data/raw/arxiv/year_month=2025-03/part-0.parquet``. Always written.
* Working store (``storage.backend: duckdb``): a single DuckDB file mirroring
  the papers plus the citation history, used for fast filtered reads.

Upserts keep one row per base arXiv ID (the latest version's metadata), so
incremental harvests can be merged repeatedly without creating duplicates.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import List, Optional

import pandas as pd

from .config import Config
from .utils import ensure_dir, get_logger

log = get_logger("storage")

STORE_COLUMNS = [
    "arxiv_id", "version", "title", "abstract", "authors", "categories",
    "primary_category", "first_submitted", "last_updated", "doi", "journal_ref",
    "license", "ingested_at", "source", "year_month",
]
HISTORY_COLUMNS = ["arxiv_id", "snapshot_date", "cited_by_count"]


def _conform(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STORE_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col not in ("version",) else 1
    out["version"] = pd.to_numeric(out["version"], errors="coerce").fillna(1).astype(int)
    for col in ("first_submitted", "last_updated"):
        out[col] = pd.to_datetime(out[col], errors="coerce", utc=True).dt.tz_localize(None)
    if not out["ingested_at"].astype(bool).any():
        out["ingested_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["year_month"] = out["first_submitted"].dt.strftime("%Y-%m")
    text_cols = [c for c in STORE_COLUMNS if c not in ("version", "first_submitted", "last_updated")]
    for col in text_cols:
        out[col] = out[col].fillna("").astype(str)
    return out[STORE_COLUMNS]


def _latest_per_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["arxiv_id", "version", "last_updated"], na_position="first")
    return df.drop_duplicates(subset=["arxiv_id"], keep="last")


class PaperStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = cfg.resolve_storage_backend()
        self.root = os.path.join(cfg.raw_dir, "arxiv")
        self.history_path = os.path.join(cfg.raw_dir, "citation_history.parquet")
        self.db_path = cfg.abspath(cfg["storage"]["path"])

    # -- papers ---------------------------------------------------------------
    def _partition_path(self, ym: str) -> str:
        return os.path.join(self.root, f"year_month={ym}", "part-0.parquet")

    def upsert(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        data = _conform(df).dropna(subset=["first_submitted"])
        for ym, part in data.groupby("year_month"):
            path = self._partition_path(str(ym))
            if os.path.exists(path):
                part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
            ensure_dir(os.path.dirname(path))
            _latest_per_id(part).to_parquet(path, index=False)
        if self.backend == "duckdb":
            self._duck_upsert(data)
        log.info("Stored %d papers (%d partitions, backend=%s).",
                 len(data), data["year_month"].nunique(), self.backend)
        return len(data)

    def read(self, start: Optional[str] = None, end: Optional[str] = None,
             columns: Optional[List[str]] = None) -> pd.DataFrame:
        if self.backend == "duckdb" and os.path.exists(self.db_path):
            return self._duck_read(start, end, columns)
        if not os.path.isdir(self.root):
            return pd.DataFrame(columns=columns or STORE_COLUMNS)
        frames = []
        for name in sorted(os.listdir(self.root)):
            ym = name.split("=", 1)[-1]
            if start and ym < start[:7]:
                continue
            if end and ym > end[:7]:
                continue
            path = os.path.join(self.root, name, "part-0.parquet")
            if os.path.exists(path):
                frames.append(pd.read_parquet(path, columns=columns))
        if not frames:
            return pd.DataFrame(columns=columns or STORE_COLUMNS)
        df = pd.concat(frames, ignore_index=True)
        if "first_submitted" in df.columns:
            if start:
                df = df[df["first_submitted"] >= pd.Timestamp(start)]
            if end:
                df = df[df["first_submitted"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
        return df.reset_index(drop=True)

    def count(self) -> int:
        return len(self.read(columns=["arxiv_id"]))

    def latest_date(self) -> Optional[pd.Timestamp]:
        df = self.read(columns=["last_updated"])
        return None if df.empty else pd.Timestamp(df["last_updated"].max())

    # -- citation history ------------------------------------------------------
    def append_citations(self, records: pd.DataFrame) -> None:
        """Record ``(arxiv_id, snapshot_date, cited_by_count)`` rows."""
        if records.empty:
            return
        rec = records[HISTORY_COLUMNS].copy()
        rec["snapshot_date"] = pd.to_datetime(rec["snapshot_date"]).dt.strftime("%Y-%m-%d")
        rec["cited_by_count"] = rec["cited_by_count"].astype(int)
        if os.path.exists(self.history_path):
            rec = pd.concat([pd.read_parquet(self.history_path), rec], ignore_index=True)
        rec = rec.drop_duplicates(subset=["arxiv_id", "snapshot_date"], keep="last")
        ensure_dir(os.path.dirname(self.history_path))
        rec.to_parquet(self.history_path, index=False)
        if self.backend == "duckdb":
            con = self._duck()
            con.execute("CREATE OR REPLACE TABLE citation_history AS SELECT * FROM read_parquet(?)",
                        [self.history_path])
            con.close()

    def citation_history(self) -> pd.DataFrame:
        if not os.path.exists(self.history_path):
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        return pd.read_parquet(self.history_path)

    # -- duckdb ------------------------------------------------------------------
    def _duck(self):
        import duckdb  # type: ignore

        ensure_dir(os.path.dirname(self.db_path))
        return duckdb.connect(self.db_path)

    def _duck_upsert(self, data: pd.DataFrame) -> None:
        con = self._duck()
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                  arxiv_id VARCHAR PRIMARY KEY, version INTEGER, title VARCHAR,
                  abstract VARCHAR, authors VARCHAR, categories VARCHAR,
                  primary_category VARCHAR, first_submitted TIMESTAMP,
                  last_updated TIMESTAMP, doi VARCHAR, journal_ref VARCHAR,
                  license VARCHAR, ingested_at VARCHAR, source VARCHAR, year_month VARCHAR)
            """)
            batch = _latest_per_id(data)  # noqa: F841  (referenced by name in SQL)
            con.execute("INSERT OR REPLACE INTO papers SELECT * FROM batch")
        finally:
            con.close()

    def _duck_read(self, start, end, columns) -> pd.DataFrame:
        cols = ", ".join(columns) if columns else "*"
        sql, params = f"SELECT {cols} FROM papers WHERE 1=1", []
        if start:
            sql += " AND first_submitted >= ?"
            params.append(pd.Timestamp(start).to_pydatetime())
        if end:
            sql += " AND first_submitted < ?"
            params.append((pd.Timestamp(end) + pd.Timedelta(days=1)).to_pydatetime())
        con = self._duck()
        try:
            return con.execute(sql, params).df()
        finally:
            con.close()
