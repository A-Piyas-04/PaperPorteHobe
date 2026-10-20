"""data/extract/extractions.parquet - one latest row per paper.

Columns: arxiv_id, input_hash, prompt_version, mode (llm | llm_abstract | heuristic | heuristic_abstract),
model, extracted_at, payload (PaperExtraction JSON), problems (verified problems JSON),
n_problems_raw, n_problems_verified, cost_usd (always 0 on free tiers), llm_attempts.

Never shipped in the bundle (full payloads); only the exploded tables are.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List

import pandas as pd

from ..config import Config
from ..utils import ensure_dir

COLUMNS = ["arxiv_id", "input_hash", "prompt_version", "mode", "model", "extracted_at", "payload",
           "problems", "n_problems_raw", "n_problems_verified", "cost_usd", "llm_attempts"]


class ExtractionStore:
    def __init__(self, cfg: Config):
        self.dir = ensure_dir(cfg.extract_dir)
        self.path = os.path.join(self.dir, "extractions.parquet")
        self.df = pd.read_parquet(self.path) if os.path.exists(self.path) else pd.DataFrame(columns=COLUMNS)
        if "llm_attempts" not in self.df.columns:
            self.df["llm_attempts"] = 0
        self._pending: List[Dict] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._reindex()

    def _reindex(self) -> None:
        self._by_id = {str(r["arxiv_id"]): r for r in self.df.to_dict("records")}

    def row(self, arxiv_id: str) -> Dict[str, Any] | None:
        return self._by_id.get(arxiv_id)

    def is_current(self, arxiv_id: str, input_hash: str, prompt_version: str, want_llm: bool) -> bool:
        r = self.row(arxiv_id)
        if r is None:
            return False
        if r["input_hash"] != input_hash or str(r["prompt_version"]) != prompt_version:
            return False
        return not (want_llm and str(r["mode"]).startswith("heuristic"))   # upgrade heuristic -> llm

    def attempts(self, arxiv_id: str) -> int:
        r = self.row(arxiv_id)
        try:
            return int(r["llm_attempts"]) if r is not None and pd.notna(r["llm_attempts"]) else 0
        except (TypeError, ValueError):
            return 0

    def add(self, **row: Any) -> None:
        row["extracted_at"] = time.time()
        row.setdefault("llm_attempts", 0)
        for key in ("payload", "problems"):
            if not isinstance(row[key], str):
                row[key] = json.dumps(row[key])
        self._pending.append(row)

    def flush(self) -> None:
        if not self._pending:
            return
        new = pd.DataFrame(self._pending, columns=COLUMNS)
        frames = [f for f in (self.df, new) if len(f)]
        self.df = (pd.concat(frames, ignore_index=True)
                   .sort_values("extracted_at").drop_duplicates("arxiv_id", keep="last")
                   .reset_index(drop=True))
        self.df["llm_attempts"] = self.df["llm_attempts"].fillna(0).astype(int)
        tmp = self.path + ".tmp"
        self.df.to_parquet(tmp, index=False)
        os.replace(tmp, self.path)
        self._pending = []
        self._reindex()

    def drop(self, ids: Iterable[str]) -> None:
        """Forget these papers (forces re-extraction on the next run)."""
        self.df = self.df[~self.df["arxiv_id"].isin(set(ids))].reset_index(drop=True)
        tmp = self.path + ".tmp"
        self.df.to_parquet(tmp, index=False)
        os.replace(tmp, self.path)
        self._reindex()

    def latest(self, ids: Iterable[str]) -> pd.DataFrame:
        return self.df[self.df["arxiv_id"].isin(set(ids))].reset_index(drop=True)
