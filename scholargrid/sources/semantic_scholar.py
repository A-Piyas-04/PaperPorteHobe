"""Semantic Scholar Graph API relevance search.

Works without a key (shared, frequently rate-limited pool). Set
``SEMANTIC_SCHOLAR_API_KEY`` (or ``S2_API_KEY``) for a dedicated limit.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from . import RateLimiter, SourcePage, http_get, make_record

NAME = "semantic_scholar"
API = "https://api.semanticscholar.org/graph/v1/paper/search"
MAX_PAGE = 100
LIMITER = RateLimiter(1.0)
_FIELDS = "title,abstract,year,publicationDate,authors,externalIds,citationCount,venue,url"


def api_key() -> str:
    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY") or ""


def fetch(phrases: List[str], cursor: Optional[int], limit: int, timeout: float,
          cfg=None, deadline: Optional[float] = None) -> SourcePage:
    offset = int(cursor or 0)
    per_page = min(limit, MAX_PAGE)
    headers = {"x-api-key": api_key()} if api_key() else {}
    LIMITER.acquire(deadline)
    data = http_get(API, {"query": phrases[0], "offset": offset, "limit": per_page,
                          "fields": _FIELDS}, timeout, headers=headers).json()
    total = data.get("total")
    records = [r for r in (_record(p) for p in data.get("data") or []) if r["title"]]
    nxt = data.get("next")
    return SourcePage(NAME, records, total, int(nxt) if nxt is not None and records else None)


def _record(p: Dict) -> Dict:
    ext = p.get("externalIds") or {}
    authors = "; ".join((a.get("name") or "").strip() for a in (p.get("authors") or [])
                        if a.get("name"))
    return make_record(NAME, p.get("paperId") or "", title=p.get("title") or "",
                       abstract=p.get("abstract") or "", authors=authors,
                       date=p.get("publicationDate") or "", year=p.get("year"),
                       doi=ext.get("DOI") or "", arxiv_id=ext.get("ArXiv") or "",
                       venue=p.get("venue") or "", cited_by_count=p.get("citationCount") or 0,
                       url=p.get("url") or "")
