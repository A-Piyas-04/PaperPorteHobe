"""Papers found by live search, kept on disk so the collection grows with use.

``data/live/`` holds ``papers.parquet`` (one row per de-duplicated paper),
``vectors.npy`` (aligned embedding rows, zero until embedded) and
``meta.json`` (the embedder fingerprint the vectors belong to). If the bundle's
embedding model changes, papers are kept and simply re-embedded on later
searches. Papers are identified across sources by DOI, arXiv ID, or normalised
title plus year (allowing a one-year gap between preprint and publication).
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import acronyms as acronym_io
from .phrase import PhraseIndex, normalize
from .sources import arxiv_from_doi, normalize_doi
from .text import base_arxiv_id, title_key
from .utils import ensure_dir, get_logger, load_json, save_json

log = get_logger("live_store")

PAPERS = "papers.parquet"
VECTORS = "vectors.npy"
META = "meta.json"
COLUMNS = ["paper_id", "title", "abstract", "authors", "date", "year", "doi", "arxiv_id",
           "venue", "primary_category", "cited_by_count", "url", "sources", "added_at", "has_vec"]
_MIN_TITLE_KEY = 12


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def record_keys(rec: Dict) -> List[str]:
    keys = []
    doi = normalize_doi(rec.get("doi"))
    if doi:
        keys.append(f"doi:{doi}")
    aid = base_arxiv_id(rec.get("arxiv_id") or "") or arxiv_from_doi(doi)
    if aid:
        keys.append(f"arxiv:{aid}")
    tk = title_key(rec.get("title") or "")
    if len(tk) >= _MIN_TITLE_KEY:
        keys.append(f"title:{tk}|{rec.get('year') or ''}")
    return keys


def _title_variants(key: str) -> List[str]:
    """``title:x|2023`` also matches ``title:x|2022`` and ``title:x|2024``."""
    if not key.startswith("title:"):
        return [key]
    base, _, year = key.rpartition("|")
    if not year.isdigit():
        return [key]
    y = int(year)
    return [key, f"{base}|{y - 1}", f"{base}|{y + 1}"]


class KeyIndex:
    """Maps every identity key of a paper to a value (row number or paper id)."""

    def __init__(self) -> None:
        self._map: Dict[str, Any] = {}

    def find(self, rec: Dict) -> Any:
        for key in record_keys(rec):
            for k in _title_variants(key):
                if k in self._map:
                    return self._map[k]
        return None

    def add(self, rec: Dict, value: Any) -> None:
        for key in record_keys(rec):
            self._map.setdefault(key, value)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> KeyIndex:
        idx = cls()
        years = pd.to_datetime(df["date"], errors="coerce").dt.year
        dois = df["doi"] if "doi" in df.columns else pd.Series("", index=df.index)
        for i, (aid, doi, title, year) in enumerate(zip(df["arxiv_id"], dois, df["title"], years)):
            idx.add({"arxiv_id": str(aid or ""), "doi": str(doi or ""), "title": str(title or ""),
                     "year": None if pd.isna(year) else int(year)}, i)
        return idx


def paper_id_for(rec: Dict) -> str:
    """Bare arXiv ID when there is one (same ID the collection uses), else a prefixed ID."""
    aid = base_arxiv_id(rec.get("arxiv_id") or "")
    if aid:
        return aid
    doi = normalize_doi(rec.get("doi"))
    if doi:
        return f"doi:{doi}"
    if rec.get("source_id"):
        return f"{rec.get('source')}:{rec['source_id']}"
    return "title:" + hashlib.sha1(title_key(rec.get("title") or "").encode()).hexdigest()[:16]


def _combine(a: Dict, b: Dict) -> Dict:
    """Merge two records of the same paper, keeping the richest values."""
    out = dict(a)
    for k in ("title", "authors", "date", "doi", "arxiv_id", "venue", "primary_category", "url"):
        if not out.get(k) and b.get(k):
            out[k] = b[k]
    if len(b.get("abstract") or "") > len(out.get("abstract") or ""):
        out["abstract"] = b["abstract"]
    out["year"] = out.get("year") or b.get("year")
    out["cited_by_count"] = max(int(out.get("cited_by_count") or 0), int(b.get("cited_by_count") or 0))
    out["sources"] = sorted(set(out.get("sources") or [out.get("source")]) |
                            set(b.get("sources") or [b.get("source")]) - {None})
    if out.get("arxiv_id"):
        out["url"] = f"https://arxiv.org/abs/{base_arxiv_id(out['arxiv_id'])}"
    return out


def merge_records(records: Iterable[Dict]) -> List[Dict]:
    """De-duplicate records from several sources (DOI, arXiv ID, title + year)."""
    merged: List[Dict] = []
    keys = KeyIndex()
    for rec in records:
        rec = {**rec, "sources": sorted(set(rec.get("sources") or [rec.get("source")]) - {None})}
        j = keys.find(rec)
        if j is None:
            keys.add(rec, len(merged))
            merged.append(rec)
        else:
            merged[j] = _combine(merged[j], rec)
            keys.add(merged[j], j)
    for rec in merged:
        rec["paper_id"] = paper_id_for(rec)
    return merged


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class LiveStore:
    def __init__(self, directory: str, fingerprint: Optional[str] = None, dim: Optional[int] = None):
        self.dir = directory
        self.fingerprint = fingerprint
        self.dim = dim
        self.lock = threading.RLock()
        self.df = pd.DataFrame(columns=COLUMNS)
        self.vectors: np.ndarray = np.zeros((0, dim or 1), dtype=np.float32)
        self.phrase = PhraseIndex([], [])
        self.keys = KeyIndex()
        self.acronyms: Dict = {}

    def __len__(self) -> int:
        return len(self.df)

    @property
    def has_vec(self) -> np.ndarray:
        return self.df["has_vec"].to_numpy(dtype=bool) if len(self.df) else np.zeros(0, dtype=bool)

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, directory: str, fingerprint: Optional[str] = None,
             dim: Optional[int] = None) -> LiveStore:
        store = cls(directory, fingerprint, dim)
        path = os.path.join(directory, PAPERS)
        if not os.path.exists(path):
            return store
        try:
            df = pd.read_parquet(path)
            meta = load_json(os.path.join(directory, META)) if os.path.exists(
                os.path.join(directory, META)) else {}
            vec_path = os.path.join(directory, VECTORS)
            vectors = np.load(vec_path) if os.path.exists(vec_path) else None
        except Exception as exc:
            log.warning("Live store at %s could not be read (%s); starting empty.", directory, exc)
            return store
        df = _typed(df)
        same_model = (meta.get("fingerprint") == fingerprint and vectors is not None
                      and len(vectors) == len(df) and (dim is None or vectors.shape[1] == dim))
        if not same_model or vectors is None:
            if len(df):
                log.info("Live store vectors belong to %s, not %s; they will be re-embedded.",
                         meta.get("fingerprint"), fingerprint)
            vectors = np.zeros((len(df), dim or 1), dtype=np.float32)
            df["has_vec"] = False
        store.df, store.vectors = df, vectors.astype(np.float32)
        store.phrase = PhraseIndex(df["title"], df["abstract"])
        for i, rec in enumerate(df.to_dict("records")):
            store.keys.add(rec, i)
        store.acronyms = acronym_io.load(directory)
        log.info("Live store: %d papers (%d embedded).", len(df), int(store.has_vec.sum()))
        return store

    def save(self) -> None:
        with self.lock:
            ensure_dir(self.dir)
            out = self.df.copy()
            out["date"] = out["date"].dt.strftime("%Y-%m-%d").fillna("")
            out.to_parquet(os.path.join(self.dir, PAPERS + ".tmp"), index=False)
            os.replace(os.path.join(self.dir, PAPERS + ".tmp"), os.path.join(self.dir, PAPERS))
            with open(os.path.join(self.dir, VECTORS + ".tmp"), "wb") as fh:
                np.save(fh, self.vectors)
            os.replace(os.path.join(self.dir, VECTORS + ".tmp"), os.path.join(self.dir, VECTORS))
            save_json({"fingerprint": self.fingerprint, "dim": self.dim, "n": len(self.df)},
                      os.path.join(self.dir, META))
            acronym_io.save(self.acronyms, self.dir)

    # -- updates -------------------------------------------------------------
    def upsert(self, records: Sequence[Dict]) -> List[int]:
        """Add new papers, enrich known ones; returns the row of every record."""
        rows: List[int] = []
        new: List[Dict] = []
        with self.lock:
            n0 = len(self.df)
            for rec in records:
                j = self.keys.find(rec)
                if j is None:
                    j = n0 + len(new)
                    new.append(rec)
                    self.keys.add(rec, j)
                elif j < n0:
                    self._enrich(j, rec)
                else:
                    new[j - n0] = _combine(new[j - n0], rec)
                rows.append(j)
            if new:
                now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                frame = _typed(pd.DataFrame([{
                    "paper_id": paper_id_for(r), "title": r.get("title", ""),
                    "abstract": r.get("abstract", ""), "authors": r.get("authors", ""),
                    "date": r.get("date", ""), "year": r.get("year"), "doi": r.get("doi", ""),
                    "arxiv_id": r.get("arxiv_id", ""), "venue": r.get("venue", ""),
                    "primary_category": r.get("primary_category", ""),
                    "cited_by_count": int(r.get("cited_by_count") or 0), "url": r.get("url", ""),
                    "sources": ";".join(r.get("sources") or [r.get("source", "")]),
                    "added_at": now, "has_vec": False} for r in new]))
                self.df = frame if not n0 else pd.concat([self.df, frame], ignore_index=True)
                self.vectors = np.vstack([self.vectors,
                                          np.zeros((len(new), self.vectors.shape[1]), dtype=np.float32)])
                self.phrase.append(frame["title"], frame["abstract"])
                self.acronyms = acronym_io.merge(self.acronyms, acronym_io.mine(frame["abstract"]))
        return rows

    def _enrich(self, j: int, rec: Dict) -> None:
        row = self.df.loc[j]
        sources = set(str(row["sources"]).split(";")) | set(rec.get("sources") or [rec.get("source")])
        self.df.at[j, "sources"] = ";".join(sorted(s for s in sources if s))
        self.df.at[j, "cited_by_count"] = max(int(row["cited_by_count"]), int(rec.get("cited_by_count") or 0))
        for col in ("doi", "arxiv_id", "venue", "authors", "primary_category"):
            if not row[col] and rec.get(col):
                self.df.at[j, col] = rec[col]
        if len(rec.get("abstract") or "") > len(row["abstract"] or ""):
            self.df.at[j, "abstract"] = rec["abstract"]
            self.phrase.abstracts[j] = f" {normalize(rec['abstract'])} "
            self.df.at[j, "has_vec"] = False

    def embed_missing(self, embedder, priority: Iterable[int] = (), limit: int = 300,
                      budget_s: Optional[float] = None, batch: int = 32) -> int:
        """Embed papers without vectors, ``priority`` rows first, stopping at
        ``limit`` papers or after ``budget_s`` seconds (whichever comes first)."""
        if embedder is None or limit <= 0:
            return 0
        with self.lock:
            missing = np.flatnonzero(~self.has_vec)
            if not len(missing):
                return 0
            pri = [int(i) for i in dict.fromkeys(priority) if 0 <= int(i) < len(self.df)
                   and not self.df.at[int(i), "has_vec"]]
            pri_set = set(pri)
            todo = (pri + [int(i) for i in missing if int(i) not in pri_set])[:limit]
            texts = (self.df["title"].iloc[todo].astype(str) + ". " +
                     self.df["abstract"].iloc[todo].fillna("").astype(str)).tolist()
        t0 = time.monotonic()
        done = 0
        for s in range(0, len(todo), batch):
            if budget_s is not None and done and time.monotonic() - t0 >= budget_s:
                break
            rows = todo[s:s + batch]
            vecs = np.asarray(embedder.transform(texts[s:s + batch]), dtype=np.float32)
            with self.lock:
                if self.vectors.shape[1] != vecs.shape[1]:
                    self.vectors = np.zeros((len(self.df), vecs.shape[1]), dtype=np.float32)
                    self.df["has_vec"] = False
                    self.dim = int(vecs.shape[1])
                self.vectors[rows] = vecs
                self.df.loc[rows, "has_vec"] = True
            done += len(rows)
        return done

    def row_record(self, j: int) -> Dict:
        r = self.df.iloc[j]
        return {**r.to_dict(), "sources": [s for s in str(r["sources"]).split(";") if s]}

    def find_row(self, paper_id: str) -> Optional[int]:
        if not len(self.df):
            return None
        hits = np.flatnonzero(self.df["paper_id"].to_numpy() == paper_id)
        return int(hits[0]) if len(hits) else None


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = False if col == "has_vec" else ("" if col not in ("year", "cited_by_count") else None)
    for col in ("paper_id", "title", "abstract", "authors", "doi", "arxiv_id", "venue",
                "primary_category", "url", "sources", "added_at"):
        df[col] = df[col].fillna("").astype(str)
    df["date"] = pd.to_datetime(df["date"].replace("", None), errors="coerce", format="mixed")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce").fillna(0).astype(int)
    df["has_vec"] = df["has_vec"].fillna(False).astype(bool)
    return df[COLUMNS]
