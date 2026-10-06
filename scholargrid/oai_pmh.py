"""arXiv OAI-PMH harvester (``metadataPrefix=arXiv``).

The official bulk interface: date-ranged (``from``/``until`` on the record
datestamp), resumable via resumption tokens, and polite (fixed delay between
requests, honours ``Retry-After`` on HTTP 503). Harvests in month-sized chunks
and reports each finished chunk to a callback, so a long backfill that dies can
resume from the last completed chunk (state in ``data/raw/oai_state.json``).
"""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .utils import get_logger, load_json, save_json

log = get_logger("oai")

_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "arxiv": "http://arxiv.org/OAI/arXiv/"}
USER_AGENT = "ScholarGrid/2.0 (research landscape explorer; mailto:{mailto})"

FetchFn = Callable[[str], bytes]


def parse_record(rec: ET.Element) -> Optional[Dict]:
    header = rec.find("oai:header", _NS)
    if header is not None and header.attrib.get("status") == "deleted":
        return None
    meta = rec.find("oai:metadata/arxiv:arXiv", _NS)
    if meta is None:
        return None

    def text(tag: str) -> str:
        node = meta.find(f"arxiv:{tag}", _NS)
        return (node.text or "").strip() if node is not None and node.text else ""

    authors = []
    for a in meta.findall("arxiv:authors/arxiv:author", _NS):
        fore = (a.findtext("arxiv:forenames", default="", namespaces=_NS) or "").strip()
        key = (a.findtext("arxiv:keyname", default="", namespaces=_NS) or "").strip()
        name = " ".join(p for p in (fore, key) if p)
        if name:
            authors.append(name)
    cats = text("categories")
    return {
        "arxiv_id": text("id"),
        "title": text("title"),
        "abstract": text("abstract"),
        "authors": "; ".join(authors),
        "categories": cats,
        "primary_category": cats.split()[0] if cats else "",
        "first_submitted": text("created"),
        "last_updated": text("updated") or text("created"),
        "doi": text("doi"),
        "journal_ref": text("journal-ref"),
        "license": text("license"),
    }


def parse_page(payload: bytes) -> Tuple[List[Dict], Optional[str]]:
    root = ET.fromstring(payload)
    err = root.find("oai:error", _NS)
    if err is not None:
        if err.attrib.get("code") == "noRecordsMatch":
            return [], None
        raise RuntimeError(f"OAI-PMH error {err.attrib.get('code')}: {err.text}")
    records = []
    for rec in root.findall("oai:ListRecords/oai:record", _NS):
        parsed = parse_record(rec)
        if parsed and parsed["arxiv_id"]:
            records.append(parsed)
    token_node = root.find("oai:ListRecords/oai:resumptionToken", _NS)
    token = (token_node.text or "").strip() if token_node is not None and token_node.text else None
    return records, token


def _http_fetch(url: str, mailto: str, max_retries: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT.format(mailto=mailto or "n/a")})
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 503 and attempt < max_retries:
                wait = int(exc.headers.get("Retry-After", "30") or 30)
                log.info("OAI-PMH asked us to wait %ss (503).", wait)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt >= max_retries:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError("unreachable")


def month_chunks(start: date, end: date) -> Iterator[Tuple[date, date]]:
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield cur, min(end, nxt - timedelta(days=1))
        cur = nxt


def harvest_range(start: date, end: date, *, endpoint: str, set_spec: str,
                  metadata_prefix: str = "arXiv", sleep_seconds: float = 5.0,
                  mailto: str = "", max_retries: int = 5,
                  fetch: Optional[FetchFn] = None) -> List[Dict]:
    """All records whose datestamp falls in [start, end]."""
    fetch = fetch or (lambda url: _http_fetch(url, mailto, max_retries))
    params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix,
              "from": start.isoformat(), "until": end.isoformat()}
    if set_spec:
        params["set"] = set_spec
    url = endpoint + "?" + urllib.parse.urlencode(params)
    out: List[Dict] = []
    while True:
        records, token = parse_page(fetch(url))
        out.extend(records)
        if not token:
            break
        url = endpoint + "?" + urllib.parse.urlencode({"verb": "ListRecords", "resumptionToken": token})
        time.sleep(sleep_seconds)
    return out


def harvest(start: date, end: date, oai_cfg: Dict, state_path: str, mailto: str = "",
            on_chunk: Optional[Callable[[List[Dict], date, date], None]] = None,
            fetch: Optional[FetchFn] = None) -> int:
    """Harvest month by month from the last completed chunk (resumable).
    Returns the number of records harvested in this call."""
    state = load_json(state_path) if os.path.exists(state_path) else {}
    done_until = state.get("completed_until")
    if done_until:
        start = max(start, date.fromisoformat(done_until) + timedelta(days=1))
    total = 0
    for lo, hi in month_chunks(start, end):
        rows = harvest_range(lo, hi, endpoint=oai_cfg["endpoint"], set_spec=oai_cfg["set"],
                             metadata_prefix=oai_cfg["metadata_prefix"],
                             sleep_seconds=float(oai_cfg["sleep_seconds"]),
                             mailto=mailto, max_retries=int(oai_cfg["max_retries"]), fetch=fetch)
        log.info("OAI-PMH %s..%s -> %d records", lo, hi, len(rows))
        if on_chunk is not None:
            on_chunk(rows, lo, hi)
        total += len(rows)
        # The current month is never "complete": keep re-harvesting it next time.
        if hi < date.today().replace(day=1):
            state["completed_until"] = hi.isoformat()
        state["last_run"] = date.today().isoformat()
        save_json(state, state_path)
    return total
