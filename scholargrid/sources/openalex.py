"""OpenAlex works search: journals, conferences and preprints across publishers."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from . import RateLimiter, SourcePage, http_get, make_record

NAME = "openalex"
API = "https://api.openalex.org/works"
MAX_PAGE = 200
LIMITER = RateLimiter(0.1)
_FIELDS = ("id,doi,title,publication_year,publication_date,cited_by_count,ids,"
           "primary_location,authorships,abstract_inverted_index,type")


def build_query(phrases: List[str]) -> str:
    out = []
    for p in phrases:
        p = p.replace('"', " ").strip()
        if p:
            out.append(f'"{p}"' if " " in p else p)
    return " OR ".join(out)


def fetch(phrases: List[str], cursor: Optional[int], limit: int, timeout: float,
          cfg=None, deadline: Optional[float] = None) -> SourcePage:
    page = int(cursor or 1)
    per_page = min(limit, MAX_PAGE)
    params = {"search": build_query(phrases), "per-page": per_page, "page": page,
              "select": _FIELDS}
    mailto = os.environ.get("OPENALEX_MAILTO") or (cfg["enrich"].get("mailto") if cfg else "")
    if mailto:
        params["mailto"] = mailto
    LIMITER.acquire(deadline)
    data = http_get(API, params, timeout).json()
    total = (data.get("meta") or {}).get("count")
    records = [r for r in (_record(w) for w in data.get("results") or []) if r["title"]]
    more = bool(records) and (total is None or page * per_page < total)
    return SourcePage(NAME, records, total, page + 1 if more else None)


def abstract_from_index(inv: Optional[Dict[str, List[int]]]) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _record(w: Dict) -> Dict:
    loc = w.get("primary_location") or {}
    source = loc.get("source") or {}
    venue = source.get("display_name") or ""
    landing = loc.get("landing_page_url") or ""
    arxiv_id = ""
    if "arxiv.org/abs/" in landing:
        arxiv_id = landing.split("arxiv.org/abs/", 1)[1]
    if "arxiv" in venue.lower():
        venue = ""
    authors = "; ".join(((a.get("author") or {}).get("display_name") or "").strip()
                        for a in (w.get("authorships") or [])
                        if (a.get("author") or {}).get("display_name"))
    oid = (w.get("id") or "").rsplit("/", 1)[-1]
    return make_record(NAME, oid, title=w.get("title") or "",
                       abstract=abstract_from_index(w.get("abstract_inverted_index")),
                       authors=authors, date=w.get("publication_date") or "",
                       year=w.get("publication_year"), doi=w.get("doi") or "",
                       arxiv_id=arxiv_id, venue=venue,
                       cited_by_count=w.get("cited_by_count") or 0,
                       url=(w.get("doi") or landing or w.get("id") or ""))
