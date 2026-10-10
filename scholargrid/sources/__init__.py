"""Live paper sources queried at search time (arXiv, OpenAlex, Semantic Scholar).

Every adapter exposes ``fetch(phrases, cursor, limit, timeout, cfg) -> SourcePage``
and returns records in one shared shape (see :func:`make_record`), so the live
search layer can merge and de-duplicate them without knowing where they came
from. ``phrases`` is the query followed by its acronym/synonym expansions.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

SOURCE_LABELS = {"arxiv": "arXiv", "openalex": "OpenAlex", "semantic_scholar": "Semantic Scholar"}


class RateLimited(RuntimeError):
    """The source answered 429; it is put on cooldown."""


class SourceUnavailable(RuntimeError):
    """The source is cooling down or its rate limiter could not admit the call in time."""


@dataclass
class SourcePage:
    source: str
    records: List[Dict] = field(default_factory=list)
    total: Optional[int] = None
    next_cursor: Any = None          # None once the source has no more results


def make_record(source: str, source_id: str, *, title: str, abstract: str = "",
                authors: str = "", date: str = "", year: Optional[int] = None,
                doi: str = "", arxiv_id: str = "", venue: str = "",
                primary_category: str = "", cited_by_count: int = 0, url: str = "") -> Dict:
    from ..text import base_arxiv_id, normalize_text

    doi = normalize_doi(doi)
    arxiv_id = base_arxiv_id(arxiv_id) if arxiv_id else arxiv_from_doi(doi)
    if not year and date[:4].isdigit():
        year = int(date[:4])
    return {
        "source": source, "source_id": str(source_id or ""),
        "title": normalize_text(title), "abstract": normalize_text(abstract),
        "authors": authors or "", "date": (date or "")[:10], "year": int(year) if year else None,
        "doi": doi, "arxiv_id": arxiv_id, "venue": venue or "",
        "primary_category": primary_category or "", "cited_by_count": int(cited_by_count or 0),
        "url": url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else
                       (f"https://doi.org/{doi}" if doi else "")),
    }


def normalize_doi(doi: Optional[str]) -> str:
    d = str(doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d


def arxiv_from_doi(doi: str) -> str:
    """arXiv's own DOIs look like ``10.48550/arxiv.2301.01234``."""
    if doi.startswith("10.48550/arxiv."):
        return doi.split("arxiv.", 1)[1]
    return ""


# ---------------------------------------------------------------------------
# Rate limiting and cooldown (shared by every session in the process)
# ---------------------------------------------------------------------------
class RateLimiter:
    """At most one call per ``min_interval`` seconds, across threads."""

    def __init__(self, min_interval: float, clock: Callable[[], float] = time.monotonic):
        self.min_interval = float(min_interval)
        self._clock = clock
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self, deadline: Optional[float] = None) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            if deadline is not None and start > deadline:
                raise SourceUnavailable("rate limiter could not admit the call before the deadline")
            self._next = start + self.min_interval
        if start > now:
            time.sleep(start - now)


_COOLDOWN: Dict[str, float] = {}
_COOLDOWN_LOCK = threading.Lock()


def cooling_down(source: str, clock: Callable[[], float] = time.monotonic) -> bool:
    with _COOLDOWN_LOCK:
        return _COOLDOWN.get(source, 0.0) > clock()


def start_cooldown(source: str, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
    with _COOLDOWN_LOCK:
        _COOLDOWN[source] = clock() + float(seconds)


def reset_cooldowns() -> None:
    with _COOLDOWN_LOCK:
        _COOLDOWN.clear()


def http_get(url: str, params: Dict[str, Any], timeout: float, headers: Optional[Dict] = None):
    import requests

    resp = requests.get(url, params=params, timeout=max(0.5, timeout), headers=headers or {})
    if resp.status_code == 429:
        raise RateLimited(f"{url} returned 429")
    resp.raise_for_status()
    return resp


def get_adapter(name: str):
    from . import arxiv, openalex, semantic_scholar

    return {"arxiv": arxiv, "openalex": openalex, "semantic_scholar": semantic_scholar}[name]
