"""Metadata for arbitrary OpenAlex works (foundations are often outside the corpus).

Cached in ``data/raw/openalex_works_cache.parquet`` (one row per work, refreshed after
``lookup_ttl_days``). In the app several sessions share one ``WorkLookup``, so cache writes are locked.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..config import Config
from ..utils import ensure_dir, get_logger

log = get_logger("foundations.lookup")
API = "https://api.openalex.org/works"
SELECT = ("id,doi,title,publication_year,cited_by_count,type,abstract_inverted_index,primary_location,"
          "locations,authorships,ids")
COLUMNS = ["openalex_id", "title", "year", "cited_by_count", "doi", "arxiv_id", "type", "venue",
           "abstract", "authors", "url", "fetched_at"]
_WRITE_LOCK = threading.Lock()


def _row(w: Dict) -> Dict:
    from ..enrich import work_arxiv_ids
    from ..sources.openalex import abstract_from_index

    loc = (w.get("primary_location") or {})
    src = loc.get("source") or {}
    arxiv = sorted(work_arxiv_ids(w))
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    names = [((a.get("author") or {}).get("display_name") or "") for a in (w.get("authorships") or [])][:6]
    oid = (w.get("id") or "").rsplit("/", 1)[-1]
    return {"openalex_id": oid, "title": w.get("title") or "", "year": w.get("publication_year"),
            "cited_by_count": int(w.get("cited_by_count") or 0), "doi": doi,
            "arxiv_id": arxiv[0] if arxiv else "", "type": w.get("type") or "",
            "venue": src.get("display_name") or "", "abstract": abstract_from_index(w.get("abstract_inverted_index")),
            "authors": "; ".join(n for n in names if n),
            "url": (f"https://arxiv.org/abs/{arxiv[0]}" if arxiv else
                    (f"https://doi.org/{doi}" if doi else f"https://openalex.org/{oid}")),
            "fetched_at": time.time()}


class WorkLookup:
    def __init__(self, cfg: Config, synthetic: bool = False):
        self.cfg = cfg
        self.f = cfg["foundations"]
        self.path = os.path.join(cfg.raw_dir, "openalex_works_cache.parquet")
        self.synthetic = synthetic
        self.df = pd.read_parquet(self.path) if os.path.exists(self.path) else pd.DataFrame(columns=COLUMNS)

    def get(self, ids: Iterable[str], deadline: Optional[float] = None) -> pd.DataFrame:
        """Rows for ``ids`` (cached or fetched). Missing works are simply absent."""
        wanted = [i for i in dict.fromkeys(ids) if i]
        if self.synthetic:
            from .synthetic import synthetic_works

            return synthetic_works(wanted)
        ttl = time.time() - float(self.f["lookup_ttl_days"]) * 86400
        have = self.df[self.df["openalex_id"].isin(wanted)]
        fresh = set(have.loc[have["fetched_at"].astype(float) >= ttl, "openalex_id"])
        todo = [i for i in wanted if i not in fresh]
        if todo:
            self._fetch(todo, deadline)
        return self.df[self.df["openalex_id"].isin(wanted)].reset_index(drop=True)

    def _fetch(self, ids: List[str], deadline: Optional[float]) -> None:
        from .. import sources
        from ..sources import RateLimited, SourceUnavailable
        from ..sources.openalex import LIMITER

        mailto = os.environ.get("OPENALEX_MAILTO") or self.cfg["enrich"].get("mailto") or ""
        rows: List[Dict] = []
        size = int(self.f["lookup_batch"])
        for i in range(0, len(ids), size):
            if deadline is not None and time.time() > deadline:
                log.info("Work lookup deadline reached; %d works left for later.", len(ids) - i)
                break
            chunk = ids[i:i + size]
            params = {"filter": "ids.openalex:" + "|".join(chunk), "per-page": str(len(chunk)), "select": SELECT}
            if mailto:
                params["mailto"] = mailto
            try:
                LIMITER.acquire(deadline)
                data = sources.http_get(API, params, float(self.f["lookup_timeout_s"])).json()
            except (RateLimited, SourceUnavailable) as exc:
                log.warning("OpenAlex lookup paused (%s).", exc)
                break
            except Exception as exc:
                log.warning("OpenAlex lookup batch failed (%s); continuing.", exc)
                continue
            rows += [_row(w) for w in data.get("results") or []]
        if rows:
            with _WRITE_LOCK:
                new = pd.DataFrame(rows, columns=COLUMNS)
                frames = [f for f in (self.df, new) if len(f)]
                self.df = (pd.concat(frames, ignore_index=True)
                           .sort_values("fetched_at").drop_duplicates("openalex_id", keep="last")
                           .reset_index(drop=True))
                ensure_dir(os.path.dirname(self.path))
                tmp = self.path + ".tmp"
                self.df.to_parquet(tmp, index=False)
                os.replace(tmp, self.path)
