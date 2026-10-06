"""OpenAlex enrichment: citations, references, venue, authors and topics.

Papers are matched by their arXiv DataCite DOI (``10.48550/arxiv.<id>``) in
batches; optionally, unmatched papers fall back to an exact normalised-title
lookup. Responses are cached in ``data/raw/openalex_cache.parquet`` and
refreshed on an age-based schedule (young papers weekly, older ones monthly or
quarterly), since citations accumulate as papers age. Every refresh appends to
the citation history so the app can show citation velocity, not just totals.
Network failures are tolerated: affected papers keep their last good values.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from .config import Config
from .text import title_key
from .utils import get_logger, load_json

log = get_logger("enrich")

_API = "https://api.openalex.org/works"
_S2_API = "https://api.semanticscholar.org/graph/v1/paper/batch"
_DOI_PREFIX = "https://doi.org/10.48550/arxiv."
_USER_AGENT = "ScholarGrid/2.0 (mailto:{mailto})"
ENRICH_COLUMNS = ["cited_by_count", "reference_count", "venue", "venue_type", "references",
                  "openalex_id", "author_ids", "topic", "citation_velocity"]
_SELECT = ("id,doi,title,cited_by_count,counts_by_year,referenced_works,primary_location,"
           "authorships,publication_date,type,primary_topic")


def enrich(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return ``df`` with ENRICH_COLUMNS added (always present)."""
    e = cfg["enrich"]
    out = df.copy()
    records: Dict[str, Optional[Dict]] = {}
    live = e["openalex"] and cfg["data"]["source"] != "synthetic" and not df.attrs.get("fallback_from")
    if live:
        records = _lookup(out, cfg)

    def field(aid: str, key: str, default):
        rec = records.get(aid)
        return rec.get(key, default) if rec else default

    ids = out["arxiv_id"].astype(str)
    out["cited_by_count"] = [int(field(a, "cited_by_count", 0)) for a in ids]
    out["references"] = [" ".join(field(a, "references", [])) for a in ids]
    out["reference_count"] = [len(field(a, "references", [])) for a in ids]
    out["venue"] = [field(a, "venue", "") for a in ids]
    out["venue_type"] = [field(a, "venue_type", "") for a in ids]
    out["openalex_id"] = [field(a, "openalex_id", "") for a in ids]
    out["author_ids"] = [" ".join(field(a, "author_ids", [])) for a in ids]
    out["topic"] = [field(a, "topic", "") for a in ids]
    out["citation_velocity"] = 0.0

    if live:
        from .storage import PaperStore

        store = PaperStore(cfg)
        matched = out[out["openalex_id"] != ""]
        store.append_citations(pd.DataFrame({
            "arxiv_id": matched["arxiv_id"], "snapshot_date": date.today().isoformat(),
            "cited_by_count": matched["cited_by_count"]}))
        velocity = citation_velocity(store.citation_history())
        out["citation_velocity"] = ids.map(velocity).fillna(0.0).astype(float)
        if e["semantic_scholar"]:
            _cross_check_s2(out)
        stats = match_stats(out)
        log.info("OpenAlex matched %d / %d papers (%.0f%%; %.0f%% of papers older than 3 months).",
                 stats["matched"], stats["papers"], 100 * stats["match_rate"],
                 100 * (stats["match_rate_older_3m"] or 0))
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def match_stats(df: pd.DataFrame) -> Dict:
    """Match and reference coverage overall, for older papers, and per month /
    category (to spot enrichment regressions)."""
    if df.empty or "openalex_id" not in df.columns:
        return {"papers": int(len(df)), "matched": 0, "match_rate": 0.0,
                "match_rate_older_3m": None, "reference_coverage_older_6m": None,
                "by_month": {}, "by_category": {}}
    matched = df["openalex_id"].astype(str) != ""
    ref = df["date"].max()
    older3 = df["date"] <= ref - pd.DateOffset(months=3)
    older6 = df["date"] <= ref - pd.DateOffset(months=6)

    def rate(mask) -> Optional[float]:
        return round(float(matched[mask].mean()), 4) if mask.any() else None

    return {
        "papers": int(len(df)),
        "matched": int(matched.sum()),
        "match_rate": round(float(matched.mean()), 4),
        "match_rate_older_3m": rate(older3),
        "reference_coverage_older_6m": (round(float((df.loc[older6, "reference_count"] > 0).mean()), 4)
                                        if older6.any() else None),
        "by_month": {str(k): round(float(v), 3) for k, v in matched.groupby(df["year_month"]).mean().items()},
        "by_category": {str(k): round(float(v), 3)
                        for k, v in matched.groupby(df["primary_category"]).mean().nlargest(40).items()},
    }


def citation_velocity(history: pd.DataFrame, horizon_days: int = 90) -> Dict[str, float]:
    """Citations gained per 30 days between each paper's oldest snapshot within
    ``horizon_days`` and its latest one. Needs two snapshots per paper."""
    if history.empty:
        return {}
    h = history.copy()
    h["snapshot_date"] = pd.to_datetime(h["snapshot_date"])
    cutoff = h["snapshot_date"].max() - pd.Timedelta(days=horizon_days)
    h = h[h["snapshot_date"] >= cutoff].sort_values("snapshot_date")
    out: Dict[str, float] = {}
    for aid, g in h.groupby("arxiv_id"):
        if len(g) < 2:
            continue
        days = (g["snapshot_date"].iloc[-1] - g["snapshot_date"].iloc[0]).days
        if days <= 0:
            continue
        gained = int(g["cited_by_count"].iloc[-1]) - int(g["cited_by_count"].iloc[0])
        out[str(aid)] = round(max(0, gained) * 30.0 / days, 3)
    return out


# ---------------------------------------------------------------------------
# Cache + refresh schedule
# ---------------------------------------------------------------------------
def refresh_interval_days(paper_date: pd.Timestamp, today: date, schedule: Dict) -> int:
    age_days = (pd.Timestamp(today) - pd.Timestamp(paper_date)).days
    if age_days < 91:
        return int(schedule["lt_3m_days"])
    if age_days < 730:
        return int(schedule["lt_24m_days"])
    return int(schedule["older_days"])


def is_stale(entry: Optional[Dict], paper_date, today: date, cfg_enrich: Dict) -> bool:
    if not entry or not entry.get("at"):
        return True
    try:
        age = (today - date.fromisoformat(str(entry["at"])[:10])).days
    except ValueError:
        return True
    if not entry.get("rec"):
        return age >= int(cfg_enrich["miss_retry_days"])
    return age >= refresh_interval_days(paper_date, today, cfg_enrich["refresh_schedule"])


class _Cache:
    """arxiv_id -> {"at": iso date, "rec": dict | None}, persisted as Parquet."""

    def __init__(self, cfg: Config):
        self.path = os.path.join(cfg.raw_dir, "openalex_cache.parquet")
        self.legacy = os.path.join(cfg.raw_dir, "openalex.json")
        self.data: Dict[str, Dict] = {}
        if os.path.exists(self.path):
            t = pd.read_parquet(self.path)
            self.data = {a: {"at": at, "rec": json.loads(r) if r else None}
                         for a, at, r in zip(t["arxiv_id"], t["fetched_at"], t["payload"])}
        elif os.path.exists(self.legacy):
            # v1 entries lack openalex_id/venue_type/topic, so all of them are refetched once.
            self.data = {a: {"at": "", "rec": (v or {}).get("rec")} for a, v in load_json(self.legacy).items()}
            log.info("Migrating %d OpenAlex cache entries from %s.", len(self.data), self.legacy)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        pd.DataFrame({
            "arxiv_id": list(self.data.keys()),
            "fetched_at": [v.get("at", "") for v in self.data.values()],
            "payload": [json.dumps(v["rec"]) if v.get("rec") else "" for v in self.data.values()],
        }).to_parquet(self.path, index=False)


def _lookup(df: pd.DataFrame, cfg: Config) -> Dict[str, Optional[Dict]]:
    e = cfg["enrich"]
    cache = _Cache(cfg)
    today = date.today()
    ids = df["arxiv_id"].astype(str).tolist()
    dates = dict(zip(ids, df["date"]))
    todo = [a for a in ids if is_stale(cache.data.get(a), dates[a], today, e)]
    batch = max(1, min(int(e["batch"]), 100))
    if todo:
        log.info("Querying OpenAlex for %d papers (%d fresh in cache).", len(todo), len(ids) - len(todo))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            found = _fetch_by_doi(chunk, e["mailto"])
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("OpenAlex batch %d failed (%s); keeping cached values.", i // batch, exc)
            continue
        for aid in chunk:
            rec = found.get(aid.lower())
            prev = (cache.data.get(aid) or {}).get("rec")
            cache.data[aid] = {"at": today.isoformat(), "rec": rec or prev}
        time.sleep(0.11)

    if e["title_fallback"]:
        titles = dict(zip(ids, df["title"].astype(str)))
        unmatched = [a for a in todo if not (cache.data.get(a) or {}).get("rec")]
        for aid in unmatched[: int(e["title_fallback_limit"])]:
            try:
                rec = _fetch_by_title(titles[aid], e["mailto"])
            except Exception as exc:  # pragma: no cover - network dependent
                log.warning("OpenAlex title lookup failed for %s: %s", aid, exc)
                continue
            if rec:
                cache.data[aid] = {"at": today.isoformat(), "rec": rec}
            time.sleep(0.11)
    if todo:
        cache.save()
    return {a: (cache.data.get(a) or {}).get("rec") for a in ids}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _get(params: Dict[str, str], mailto: str) -> Dict:
    if mailto:
        params["mailto"] = mailto
    url = _API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT.format(mailto=mailto or "n/a")})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt * 2)
                continue
            raise
    raise RuntimeError("unreachable")


def parse_work(w: Dict) -> Dict:
    source = ((w.get("primary_location") or {}).get("source") or {})
    venue = source.get("display_name") or ""
    venue_type = source.get("type") or ""
    if "arxiv" in venue.lower():
        venue, venue_type = "", "arxiv-only"
    authors = [((a.get("author") or {}).get("id") or "").rsplit("/", 1)[-1]
               for a in (w.get("authorships") or [])]
    return {
        "openalex_id": (w.get("id") or "").rsplit("/", 1)[-1],
        "cited_by_count": int(w.get("cited_by_count") or 0),
        "counts_by_year": {str(c.get("year")): int(c.get("cited_by_count") or 0)
                           for c in (w.get("counts_by_year") or [])},
        "references": [r.rsplit("/", 1)[-1] for r in (w.get("referenced_works") or [])],
        "venue": venue,
        "venue_type": venue_type,
        "author_ids": [a for a in authors if a],
        "publication_date": w.get("publication_date") or "",
        "type": w.get("type") or "",
        "topic": ((w.get("primary_topic") or {}).get("display_name") or ""),
    }


def _fetch_by_doi(chunk: List[str], mailto: str) -> Dict[str, Dict]:
    doi_filter = "|".join(f"{_DOI_PREFIX}{a.lower()}" for a in chunk)
    data = _get({"filter": f"doi:{doi_filter}", "per-page": str(len(chunk)), "select": _SELECT}, mailto)
    out: Dict[str, Dict] = {}
    for w in data.get("results", []):
        doi = (w.get("doi") or "").lower()
        if doi.startswith(_DOI_PREFIX):
            out[doi[len(_DOI_PREFIX):]] = parse_work(w)
    return out


def _fetch_by_title(title: str, mailto: str) -> Optional[Dict]:
    key = title_key(title)
    if len(key) < 12:
        return None
    query = key.replace(",", " ")
    data = _get({"filter": f"title.search:{query}", "per-page": "5", "select": _SELECT}, mailto)
    for w in data.get("results", []):
        if title_key(w.get("title") or "") == key:
            return parse_work(w)
    return None


def _cross_check_s2(df: pd.DataFrame, batch: int = 400) -> None:
    """Optional Semantic Scholar cross-check; logs large citation disagreements."""
    key = os.environ.get("S2_API_KEY", "")
    if not key:
        log.warning("enrich.semantic_scholar is on but S2_API_KEY is not set; skipping.")
        return
    ids = df["arxiv_id"].astype(str).tolist()
    s2: Dict[str, int] = {}
    for i in range(0, len(ids), batch):
        body = json.dumps({"ids": [f"ARXIV:{a}" for a in ids[i:i + batch]]}).encode()
        req = urllib.request.Request(_S2_API + "?fields=citationCount,externalIds", data=body,
                                     headers={"x-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for item in json.loads(resp.read().decode()):
                    if item and (item.get("externalIds") or {}).get("ArXiv"):
                        s2[item["externalIds"]["ArXiv"]] = int(item.get("citationCount") or 0)
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("Semantic Scholar batch failed: %s", exc)
        time.sleep(1.0)
    oa = dict(zip(df["arxiv_id"].astype(str), df["cited_by_count"]))
    big = [(a, oa[a], c) for a, c in s2.items() if a in oa and abs(oa[a] - c) > max(10, 0.5 * max(oa[a], c))]
    log.info("Semantic Scholar: %d papers compared, %d large disagreements.", len(s2), len(big))
    for a, o, c in big[:20]:
        log.info("  %s OpenAlex=%d S2=%d", a, o, c)
