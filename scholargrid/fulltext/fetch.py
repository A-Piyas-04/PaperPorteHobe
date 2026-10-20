"""HTTP fetcher for arXiv full text. One shared rate limiter (>= 3 s between requests).

Keep **one** ``Fetcher`` per run so all requests share its limiter; arXiv asks automated clients for
at most one request every 3 seconds on a single connection.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ..config import Config
from ..sources import RateLimiter
from ..utils import get_logger
from .model import Section
from .parse_html import is_latexml, parse_html
from .parse_pdf import parse_pdf

log = get_logger("fulltext.fetch")

URLS = {
    "arxiv_html": "https://arxiv.org/html/{id}",
    "ar5iv": "https://ar5iv.labs.arxiv.org/html/{id}",
    "pdf": "https://arxiv.org/pdf/{id}",
}


@dataclass
class FetchResult:
    status: str                       # ok | unavailable | error
    source: str = ""
    http_status: Optional[int] = None
    sections: List[Section] = field(default_factory=list)
    error: str = ""


class Fetcher:
    def __init__(self, cfg: Config, session: Any = None):
        import requests

        f = cfg["fulltext"]
        self.cfg = cfg
        self.order: List[str] = list(f["order"])
        self.keep = set(f["keep_kinds"])
        self.max_chars = int(f["max_section_chars"])
        self.max_paper_chars = int(f["max_paper_chars"])
        self.timeout = float(f["timeout_s"])
        self.retries = int(f["max_retries"])
        self.max_bytes = int(float(f["max_download_mb"]) * 1024 * 1024)
        self.limiter = RateLimiter(float(f["request_interval_s"]))
        mailto = os.environ.get("OPENALEX_MAILTO") or cfg["enrich"].get("mailto") or "n/a"
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": f"ScholarGrid/2.1 (research-starter; mailto:{mailto})"})

    def _get(self, url: str) -> Tuple[Optional[int], bytes, str]:
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            self.limiter.acquire()
            try:
                resp = self.session.get(url, timeout=self.timeout, allow_redirects=True, stream=True)
                if resp.status_code in (429, 503):
                    try:
                        wait = float(resp.headers.get("Retry-After") or 10 * (attempt + 1))
                    except ValueError:
                        wait = 10.0 * (attempt + 1)
                    log.warning("%s -> %s; backing off %.0fs", url, resp.status_code, wait)
                    time.sleep(min(wait, 120))
                    continue
                content = resp.raw.read(self.max_bytes + 1, decode_content=True) if resp.ok else b""
                if len(content) > self.max_bytes:
                    return resp.status_code, b"", "too large"
                return resp.status_code, content, ""
            except Exception as exc:   # network errors: retry with backoff
                last_exc = exc
                time.sleep(2 ** attempt)
        return None, b"", f"{type(last_exc).__name__}: {last_exc}" if last_exc else "retries exhausted"

    def _cap(self, sections: List[Section]) -> List[Section]:
        """Enforce ``max_paper_chars`` across sections (gap-relevant kinds keep priority by order)."""
        budget = self.max_paper_chars
        for s in sections:
            if len(s.text) > budget:
                s.text = s.text[:max(0, budget)]
            budget -= len(s.text)
        return sections

    def fetch(self, arxiv_id: str) -> FetchResult:
        errors = []
        for source in self.order:
            url = URLS[source].format(id=arxiv_id)
            code, body, err = self._get(url)
            if err:
                errors.append(f"{source}: {err}")
                continue
            if code == 404 or code is None or not body:
                continue
            if code >= 400:
                errors.append(f"{source}: HTTP {code}")
                continue
            try:
                if source in ("arxiv_html", "ar5iv"):
                    html = body.decode("utf-8", errors="replace")
                    if not is_latexml(html):       # e.g. ar5iv redirected to the abs page
                        continue
                    sections = parse_html(html, self.keep, self.max_chars)
                else:
                    sections = parse_pdf(body, self.cfg["fulltext"]["pdf_backend"], self.keep, self.max_chars)
            except Exception as exc:
                errors.append(f"{source}: parse {type(exc).__name__}: {exc}")
                continue
            if sum(len(s.text) for s in sections) < 500:     # parsed but empty -> try the next source
                errors.append(f"{source}: too little text")
                continue
            return FetchResult("ok", source, code, self._cap(sections))
        return FetchResult("error" if errors else "unavailable", "", None, [], "; ".join(errors)[:500])
