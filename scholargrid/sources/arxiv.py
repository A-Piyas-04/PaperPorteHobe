"""arXiv API: title/abstract phrase search, sorted by relevance."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from . import RateLimiter, SourcePage, http_get, make_record

NAME = "arxiv"
API = "https://export.arxiv.org/api/query"
MAX_PAGE = 2000
# arXiv asks for no more than one request every 3 seconds.
LIMITER = RateLimiter(3.0)
_TOTAL_RE = re.compile(r"totalResults[^>]*>(\d+)<")


def build_query(phrases: List[str]) -> str:
    parts = []
    for p in phrases:
        p = p.replace('"', " ").strip()
        if not p:
            continue
        if " " in p:
            parts.append(f'ti:"{p}" OR abs:"{p}"')
        else:
            parts.append(f"ti:{p} OR abs:{p}")
    return " OR ".join(parts)


def fetch(phrases: List[str], cursor: Optional[int], limit: int, timeout: float,
          cfg=None, deadline: Optional[float] = None) -> SourcePage:
    from ..data_ingest import _ATOM, parse_atom_entry

    start = int(cursor or 0)
    LIMITER.acquire(deadline)
    resp = http_get(API, {"search_query": build_query(phrases), "start": start,
                          "max_results": min(limit, MAX_PAGE), "sortBy": "relevance",
                          "sortOrder": "descending"}, timeout)
    m = _TOTAL_RE.search(resp.text)
    total = int(m.group(1)) if m else None
    feed = ET.fromstring(resp.content)
    records = [_record(parse_atom_entry(e)) for e in feed.findall("a:entry", _ATOM)]
    records = [r for r in records if r["title"]]
    nxt = start + len(records)
    more = bool(records) and (total is None or nxt < total)
    return SourcePage(NAME, records, total, nxt if more else None)


def _record(e: Dict) -> Dict:
    return make_record(NAME, e["arxiv_id"], title=e["title"], abstract=e["abstract"],
                       authors=e["authors"], date=e["first_submitted"], doi=e["doi"],
                       arxiv_id=e["arxiv_id"], venue=e.get("journal_ref", ""),
                       primary_category=e["primary_category"])
