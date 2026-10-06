"""Optional OpenAlex enrichment: citation counts, references and venue.

arXiv papers are looked up in OpenAlex by their DataCite DOI
(``10.48550/arxiv.<id>``) in batches. Results are cached under ``data/raw`` so
reruns stay offline. Network failures and papers OpenAlex has not indexed yet
are tolerated: they simply get zero counts and no references.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from .config import Config
from .utils import get_logger, load_json, save_json

log = get_logger("enrich")

_API = "https://api.openalex.org/works"
_DOI_PREFIX = "https://doi.org/10.48550/arxiv."
ENRICH_COLUMNS = ["cited_by_count", "reference_count", "venue", "references"]
_HIT_TTL_DAYS = 7
_MISS_TTL_DAYS = 1


def enrich(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return ``df`` with the ENRICH_COLUMNS added (always present)."""
    e = cfg.get("enrich", {}) or {}
    out = df.copy()
    records: Dict[str, Optional[Dict]] = {}

    if e.get("openalex", False) and cfg["data"]["source"] != "synthetic":
        records = _lookup(out["arxiv_id"].astype(str).tolist(), cfg)

    def field(aid: str, key: str, default):
        rec = records.get(aid)
        return rec.get(key, default) if rec else default

    ids = out["arxiv_id"].astype(str)
    out["cited_by_count"] = [int(field(a, "cited_by_count", 0)) for a in ids]
    out["references"] = [" ".join(field(a, "references", [])) for a in ids]
    out["reference_count"] = [len(field(a, "references", [])) for a in ids]
    out["venue"] = [field(a, "venue", "") for a in ids]
    hit = sum(1 for a in ids if records.get(a))
    if records:
        log.info("OpenAlex matched %d / %d papers.", hit, len(out))
    return out


def _stale(entry, today: date) -> bool:
    """Misses are retried after a day, hits refreshed weekly (citations grow)."""
    if not isinstance(entry, dict) or "at" not in entry:
        return True
    try:
        age = (today - date.fromisoformat(entry["at"])).days
    except ValueError:
        return True
    return age >= (_HIT_TTL_DAYS if entry.get("rec") else _MISS_TTL_DAYS)


def _lookup(arxiv_ids: List[str], cfg: Config) -> Dict[str, Optional[Dict]]:
    e = cfg["enrich"]
    cache_path = os.path.join(cfg.raw_dir, "openalex.json")
    cache: Dict[str, Dict] = load_json(cache_path) if os.path.exists(cache_path) else {}
    today = date.today()

    todo = [a for a in arxiv_ids if _stale(cache.get(a), today)]
    batch = max(1, min(int(e.get("batch", 50)), 100))
    if todo:
        log.info("Querying OpenAlex for %d papers (%d cached).", len(todo), len(arxiv_ids) - len(todo))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            found = _fetch(chunk, e.get("mailto") or "")
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("OpenAlex batch %d failed (%s); continuing without it.", i // batch, exc)
            continue
        for aid in chunk:
            cache[aid] = {"at": today.isoformat(), "rec": found.get(aid.lower())}
        time.sleep(0.2)
    if todo:
        save_json(cache, cache_path)
    return {a: (cache.get(a) or {}).get("rec") for a in arxiv_ids}


def _fetch(chunk: List[str], mailto: str) -> Dict[str, Dict]:
    doi_filter = "|".join(f"{_DOI_PREFIX}{a.lower()}" for a in chunk)
    params = {
        "filter": f"doi:{doi_filter}",
        "per-page": str(len(chunk)),
        "select": "doi,cited_by_count,referenced_works,primary_location",
    }
    if mailto:
        params["mailto"] = mailto
    url = _API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "ScholarGrid/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    out: Dict[str, Dict] = {}
    for w in data.get("results", []):
        doi = (w.get("doi") or "").lower()
        if not doi.startswith(_DOI_PREFIX):
            continue
        aid = doi[len(_DOI_PREFIX):]
        source = ((w.get("primary_location") or {}).get("source") or {})
        venue = source.get("display_name") or ""
        if "arxiv" in venue.lower():
            venue = ""
        out[aid] = {
            "cited_by_count": int(w.get("cited_by_count") or 0),
            "references": [r.rsplit("/", 1)[-1] for r in (w.get("referenced_works") or [])],
            "venue": venue,
        }
    return out
