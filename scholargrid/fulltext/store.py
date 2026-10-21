"""On-disk full-text store (never shipped in the app bundle).

data/fulltext/status.parquet          one row per paper: status, source, fetched_at, counts, error
data/fulltext/sections/part-*.parquet appended in flushes; latest fetch per paper wins
data/fulltext/cues/part-*.parquet     gap-cue sentences (same flush cadence)
"""
from __future__ import annotations

import glob
import os
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..config import Config
from ..utils import ensure_dir
from .model import Cue, Section

STATUS_COLUMNS = ["arxiv_id", "status", "source", "fetched_at", "http_status", "n_sections", "n_chars", "error"]


def _safe(value):
    """Make parsed text safe to write to parquet.

    HTML and PDF extraction can leave unpaired UTF-16 surrogates in a Python str (a mangled
    mathematical symbol, say). pyarrow refuses to encode those, and because it fails inside
    ``flush`` it kills the whole run and loses the buffered batch. Round-tripping through
    UTF-16 with ``surrogatepass`` rebuilds real characters from valid pairs and drops the
    unpaired remainder, so one bad paper can no longer stop a long fetch.
    """
    if not isinstance(value, str) or not any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        return value
    return value.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")


class FulltextStore:
    def __init__(self, cfg: Config):
        self.dir = ensure_dir(cfg.fulltext_dir)
        self.sections_dir = ensure_dir(os.path.join(self.dir, "sections"))
        self.cues_dir = ensure_dir(os.path.join(self.dir, "cues"))
        self.status_path = os.path.join(self.dir, "status.parquet")
        self.status = (pd.read_parquet(self.status_path) if os.path.exists(self.status_path)
                       else pd.DataFrame(columns=STATUS_COLUMNS))
        self._sec_rows: List[Dict] = []
        self._cue_rows: List[Dict] = []
        self._status_rows: List[Dict] = []
        self._last_stamp = 0

    # -- writes -------------------------------------------------------------------------
    def add(self, arxiv_id: str, status: str, source: str, sections: List[Section], cues: List[Cue],
            http_status: Optional[int] = None, error: str = "") -> None:
        now = time.time()
        self._status_rows.append({"arxiv_id": arxiv_id, "status": status, "source": source, "fetched_at": now,
                                  "http_status": http_status, "n_sections": len(sections),
                                  "n_chars": sum(len(s.text) for s in sections), "error": _safe(error)})
        for s in sections:
            row = {k: _safe(v) for k, v in s.to_row(arxiv_id, source).items()}
            self._sec_rows.append({**row, "fetched_at": now})
        for c in cues:
            self._cue_rows.append({"arxiv_id": arxiv_id, "section_idx": c.section_idx, "kind": c.kind,
                                   "sentence": _safe(c.sentence), "cue": _safe(c.cue),
                                   "char_start": c.char_start, "fetched_at": now})

    def _stamp(self) -> str:
        stamp = max(int(time.time() * 1000), self._last_stamp + 1)   # unique even within one millisecond
        self._last_stamp = stamp
        return str(stamp)

    def flush(self) -> None:
        stamp = self._stamp()
        if self._sec_rows:
            pd.DataFrame(self._sec_rows).to_parquet(os.path.join(self.sections_dir, f"part-{stamp}.parquet"),
                                                    index=False)
        if self._cue_rows:
            pd.DataFrame(self._cue_rows).to_parquet(os.path.join(self.cues_dir, f"part-{stamp}.parquet"),
                                                    index=False)
        if self._status_rows:
            new = pd.DataFrame(self._status_rows)
            frames = [f for f in (self.status, new) if len(f)]
            self.status = (pd.concat(frames, ignore_index=True)
                           .sort_values("fetched_at").drop_duplicates("arxiv_id", keep="last")
                           .reset_index(drop=True))
            tmp = self.status_path + ".tmp"
            self.status.to_parquet(tmp, index=False)
            os.replace(tmp, self.status_path)          # atomic on POSIX and Windows
        self._sec_rows, self._cue_rows, self._status_rows = [], [], []

    # -- reads --------------------------------------------------------------------------
    def needs_fetch(self, ids: Iterable[str], refetch_failed_days: int) -> List[str]:
        st = self.status.set_index("arxiv_id") if len(self.status) else pd.DataFrame()
        cutoff = time.time() - refetch_failed_days * 86400
        out = []
        for aid in ids:
            if aid not in st.index:
                out.append(aid)
                continue
            row = st.loc[aid]
            if row["status"] in ("error", "unavailable") and float(row["fetched_at"]) < cutoff:
                out.append(aid)
        return out

    def _read_parts(self, directory: str, ids: Optional[set] = None) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(directory, "part-*.parquet")))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if ids is not None:
            df = df[df["arxiv_id"].isin(ids)]
        if df.empty:
            return df.reset_index(drop=True)
        latest = df.groupby("arxiv_id")["fetched_at"].transform("max")
        return df[df["fetched_at"] == latest].reset_index(drop=True)

    def sections(self, ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return self._read_parts(self.sections_dir, set(ids) if ids is not None else None)

    def cues(self, ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return self._read_parts(self.cues_dir, set(ids) if ids is not None else None)

    def compact(self) -> None:
        """Merge part files (latest fetch per paper) into one part each."""
        for d in (self.sections_dir, self.cues_dir):
            df = self._read_parts(d)
            files = glob.glob(os.path.join(d, "part-*.parquet"))
            if df.empty:
                continue
            path = os.path.join(d, f"part-{self._stamp()}-compact.parquet")
            df.to_parquet(path, index=False)
            for f in files:
                os.remove(f)
