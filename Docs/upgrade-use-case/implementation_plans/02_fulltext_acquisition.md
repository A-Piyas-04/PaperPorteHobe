# Plan 02: Full-text acquisition and section parsing (new stage `fulltext`)

> **Prerequisites:** plan 01 is merged (`scope`, `llm`, the new dependencies). `pip install -e ".[fulltext]"`.
> **Produces:** a resumable pipeline stage that fetches the full text of in-scope arXiv papers (arXiv HTML → ar5iv → PDF), splits it into sections, classifies each section (limitations, future work, conclusion, …), finds "gap cue" sentences, and stores everything under `data/fulltext/`. A synthetic full-text generator lets CI exercise the whole chain offline.
> **Why:** authors state open problems in **Limitations / Future Work / Discussion**, not in abstracts. Without full text the gap engine (plan 06) has nothing to stand on.

---

## 1. Prompt to paste into Cursor

```text
Implement plan file 02_fulltext_acquisition.md. Follow 00_INDEX_READ_FIRST.md conventions and
the contracts from plan 01. Order: §3 config → §4 package skeleton → §5 section classifier + cue
finder → §6 HTML parser → §7 PDF parser → §8 fetcher → §9 synthetic generator → §10 store →
§11 run_fulltext + pipeline stage → §12 CLI → §13 tests. Respect arXiv's rate limits exactly as
written (one request every 3 seconds, single connection). Run ruff, mypy and pytest after each section.
Finish with the commands in §14.
```

---

## 2. Design summary

```
select_scope(enriched df)                     (plan 01: which papers get depth)
   │
   ├─ already fetched & fresh? ──► skip
   │
   ▼
fetch order (config):  arxiv_html ─404/no-article─► ar5iv ─not converted─► pdf
   │                     https://arxiv.org/html/<id>
   │                     https://ar5iv.labs.arxiv.org/html/<id>
   │                     https://arxiv.org/pdf/<id>
   ▼
parse → [Section(idx, parent_idx, heading, kind, text)]  + cue sentences
   │
   ▼
FulltextStore (data/fulltext/): status.parquet, sections/part-*.parquet, cues/part-*.parquet
```

- **Polite access:** arXiv asks automated clients to make **no more than one request every 3 seconds** and to use a single connection. Default `request_interval_s: 3.0`, one worker. At this rate 1,500 papers take ~75 minutes and 12,000 take ~10 hours. That is too long for a laptop, so real backfills run **in GitHub Actions** (the `dev-build` workflow from plan 01 §8.2, later the daily workflow from plan 11 §9.2), capped by `max_papers_per_run` and resumed run after run. On the laptop, only the synthetic source and ≈ 20-paper checks with `configs/domain_nlp_tiny.yaml`. arXiv's bulk data (requester-pays S3) is paid and deferred (`LATER_NOT_NOW.md`).
- **What is stored:** full text of "gap-relevant" kinds only (abstract, introduction, experiments, results, discussion, limitations, future work, threats to validity, broader impact, conclusion), each capped at `max_section_chars`. Other sections keep only their heading. **Full text never goes into the published bundle.** Only short verified quotes do (plan 03).
- **Licensing:** arXiv papers carry individual licenses (the `license` column from ingest). Text is used for analysis; the app shows only short quotes (≤ 300 chars) with attribution and a link to the source. See plan 11 §9.

---

## 3. Config

### 3.1 `scholargrid/schema.py`

```python
SectionKindName = Literal[
    "abstract", "introduction", "background", "related_work", "method", "experiments", "results",
    "discussion", "limitations", "future_work", "threats_to_validity", "broader_impact", "ethics",
    "conclusion", "appendix", "acknowledgements", "references", "other"]
FulltextSource = Literal["arxiv_html", "ar5iv", "pdf"]


class Fulltext(_Section):
    enabled: bool = True
    source: Literal["arxiv", "synthetic", "none"] = "arxiv"   # synthetic: generated, offline (CI/demo)
    order: List[FulltextSource] = Field(default_factory=lambda: ["arxiv_html", "ar5iv", "pdf"])
    pdf_backend: Literal["pypdf"] = "pypdf"                   # pymupdf (AGPL) is deferred: LATER_NOT_NOW.md
    request_interval_s: float = Field(default=3.0, ge=3.0)    # arXiv: max 1 request / 3 s
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=6)
    max_download_mb: float = Field(default=30.0, gt=0)
    store_dir: str = "data/fulltext"
    keep_kinds: List[SectionKindName] = Field(default_factory=lambda: [
        "abstract", "introduction", "experiments", "results", "discussion", "limitations",
        "future_work", "threats_to_validity", "broader_impact", "conclusion"])
    max_section_chars: int = Field(default=8000, ge=500)
    max_paper_chars: int = Field(default=40000, ge=2000)
    refetch_failed_days: int = Field(default=30, ge=1)
    flush_every: int = Field(default=50, ge=1)
    max_papers_per_run: Optional[int] = Field(default=None, ge=1)   # spread a big backfill over several CI runs
```

Add `fulltext: Fulltext = Field(default_factory=Fulltext)` to `ConfigModel`.

### 3.2 `configs/config.yaml`: append

```yaml
# ---------------------------------------------------------------------------
# Full text (plan 02): fetched for papers in `scope` only. Authors state open
# problems in Limitations / Future Work, which abstracts rarely contain.
# arXiv asks for at most one request every 3 seconds - do not lower that.
# ---------------------------------------------------------------------------
fulltext:
  enabled: true
  source: arxiv                  # arxiv | synthetic (offline demo/CI) | none
  order: [arxiv_html, ar5iv, pdf]
  pdf_backend: pypdf
  request_interval_s: 3.0
  timeout_s: 30
  max_retries: 3
  max_download_mb: 30
  store_dir: data/fulltext
  keep_kinds: [abstract, introduction, experiments, results, discussion, limitations,
               future_work, threats_to_validity, broader_impact, conclusion]
  max_section_chars: 8000
  max_paper_chars: 40000
  refetch_failed_days: 30
  flush_every: 50
  max_papers_per_run: null
```

`configs/ci.yaml`: `fulltext: {source: synthetic}`.
`tests/conftest.py::make_config`: `raw["fulltext"]["source"] = "synthetic"` and `raw["fulltext"]["store_dir"] = str(tmp_path / "fulltext")`.

### 3.3 `scholargrid/config.py`

Add a property:

```python
    @property
    def fulltext_dir(self) -> str:
        return self.abspath(self.raw["fulltext"]["store_dir"])
```

---

## 4. Package skeleton: `scholargrid/fulltext/`

```
scholargrid/fulltext/
  __init__.py      # exports: Section, SectionKind, FulltextStore, run_fulltext, classify_heading, find_cues
  model.py         # Section / Cue dataclasses, SectionKind literal
  sections.py      # heading classifier, cue-sentence finder, sentence splitter, text cleanup
  parse_html.py    # LaTeXML (arXiv HTML / ar5iv) -> sections
  parse_pdf.py     # PDF bytes -> sections (pypdf)
  fetch.py         # polite HTTP fetcher with source fallback order
  synthetic.py     # deterministic full text for the synthetic corpus
  store.py         # FulltextStore: status + sections + cues (Parquet parts)
  run.py           # run_fulltext(df, cfg, progress) -> stats
  __main__.py      # CLI: stats | show <id> | fetch <id...> | compact
```

### 4.1 `model.py`

```python
"""Data shapes shared by the full-text parsers and the store."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

SectionKind = Literal[
    "abstract", "introduction", "background", "related_work", "method", "experiments", "results",
    "discussion", "limitations", "future_work", "threats_to_validity", "broader_impact", "ethics",
    "conclusion", "appendix", "acknowledgements", "references", "other"]

GAP_KINDS = {"limitations", "future_work", "discussion", "conclusion", "threats_to_validity",
             "broader_impact"}


@dataclass
class Section:
    idx: int
    parent_idx: Optional[int]
    heading: str
    kind: str
    text: str
    anchor: str = ""          # HTML id (e.g. "S5") so the app can deep-link https://arxiv.org/html/<id>#S5

    def to_row(self, arxiv_id: str, source: str) -> dict:
        return {"arxiv_id": arxiv_id, "source": source, **asdict(self), "n_chars": len(self.text)}


@dataclass
class Cue:
    section_idx: int
    kind: str
    sentence: str
    cue: str                  # name of the pattern that matched
    char_start: int
```

---

## 5. `sections.py`: heading classifier, cue finder, cleanup

```python
"""Section-kind classification and gap-cue sentence detection (no ML; fast, transparent)."""
from __future__ import annotations

import re
from typing import List, Tuple

from .model import Cue, Section

_NUMBERING = re.compile(r"^\s*(?:(?:\d+|[A-Z])(?:\.\d+)*\.?|[IVXLC]+\.)\s+")

# First match wins. Order matters: "Conclusion and Future Work" -> future_work,
# "Limitations and Future Work" -> limitations.
_KIND_RULES: List[Tuple[str, re.Pattern]] = [
    ("references", re.compile(r"^(references|bibliography)$", re.I)),
    ("acknowledgements", re.compile(r"acknowledg", re.I)),
    ("limitations", re.compile(r"\blimitation", re.I)),
    ("threats_to_validity", re.compile(r"threats?\s+to\s+(the\s+)?validity", re.I)),
    ("future_work", re.compile(r"future\s+(work|direction|research)|open\s+(problem|question|challenge)s?|"
                               r"outlook|next\s+steps|research\s+agenda", re.I)),
    ("broader_impact", re.compile(r"broader\s+impact|societal\s+impact|impact\s+statement", re.I)),
    ("ethics", re.compile(r"ethic", re.I)),
    ("conclusion", re.compile(r"conclu|concluding|final\s+remarks|^summary$", re.I)),
    ("discussion", re.compile(r"discussion|lessons\s+learn", re.I)),
    ("appendix", re.compile(r"appendix|supplementa", re.I)),
    ("related_work", re.compile(r"related\s+work|prior\s+work|literature\s+review|previous\s+work", re.I)),
    ("background", re.compile(r"background|preliminar|problem\s+(formulation|statement|definition|setup)",
                              re.I)),
    ("introduction", re.compile(r"introduction|^overview$|^motivation$", re.I)),
    ("results", re.compile(r"\bresults?\b|\bfindings\b|\bevaluation\b|\banalysis\b|ablation", re.I)),
    ("experiments", re.compile(r"experiment|setup|implementation\s+details|empirical|benchmark|dataset",
                               re.I)),
    ("method", re.compile(r"method|approach|model|framework|architecture|proposed|algorithm|design",
                          re.I)),
]


def clean_heading(text: str) -> str:
    return _NUMBERING.sub("", " ".join(str(text).split())).strip(" .:")


def classify_heading(heading: str, parent_kind: str | None = None) -> str:
    h = clean_heading(heading)
    for kind, rx in _KIND_RULES:
        if rx.search(h):
            return kind
    return parent_kind or "other"


# --- sentences ---------------------------------------------------------------------------
_ABBREV = re.compile(r"\b(e\.g|i\.e|et al|cf|vs|Fig|Figs|Eq|Eqs|Sec|Tab|resp|approx|no)\.$", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def split_sentences(text: str) -> List[Tuple[int, str]]:
    """[(char_start, sentence)]. Joins splits that end in a common abbreviation."""
    out: List[Tuple[int, str]] = []
    pos, buf_start, buf = 0, 0, ""
    for part in _SPLIT.split(text):
        start = text.find(part, pos)
        pos = start + len(part)
        if buf and _ABBREV.search(buf):
            buf = buf + " " + part
            continue
        if buf:
            out.append((buf_start, buf.strip()))
        buf, buf_start = part, start
    if buf.strip():
        out.append((buf_start, buf.strip()))
    return out


# --- gap cues ------------------------------------------------------------------------------
CUE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("future_work", re.compile(r"\bfuture\s+(work|research|stud(y|ies)|direction)", re.I)),
    ("left_for_future", re.compile(r"\b(leave|left)\b[^.]{0,40}\b(for|to)\s+future", re.I)),
    ("we_plan", re.compile(r"\bwe\s+(plan|intend|hope)\s+to\b", re.I)),
    ("limitation", re.compile(r"\b(a|one|the|main|key|major|important|primary|current)\s+"
                              r"(main\s+|key\s+|major\s+)?limitations?\b", re.I)),
    ("limited_to", re.compile(r"\b(is|are|was|were|remains?)\s+limited\s+to\b", re.I)),
    ("cannot", re.compile(r"\b(our|this|the proposed)\s+(method|approach|model|system|work|framework)\s+"
                          r"(does\s+not|cannot|can\s+not|is\s+unable\s+to|fails?\s+to)\b", re.I)),
    ("open_question", re.compile(r"\b(remains?|is\s+still|are\s+still|stays?)\s+(an?\s+)?"
                                 r"(open|unclear|unsolved|unresolved|unexplored|elusive|challenging)", re.I)),
    ("open_problem", re.compile(r"\bopen\s+(problem|question|challenge|issue)s?\b", re.I)),
    ("not_explored", re.compile(r"\b(has|have)\s+not\s+(yet\s+)?been\s+(explored|studied|investigated|"
                                r"addressed|examined|evaluated)", re.I)),
    ("little_work", re.compile(r"\b(little|no|limited|scarce)\s+(prior\s+)?(work|attention|research|"
                               r"studies)\s+(has|have)\b", re.I)),
    ("out_of_scope", re.compile(r"\bbeyond\s+the\s+scope\b|\bout\s+of\s+(the\s+)?scope\b", re.I)),
    ("direction", re.compile(r"\b(interesting|promising|important|exciting|natural)\s+"
                             r"(direction|avenue|extension|next\s+step)", re.I)),
    ("does_not_generalize", re.compile(r"\b(may|might|does|do)\s+not\s+(generali[sz]e|transfer|scale|hold)\b",
                                       re.I)),
]
_CUE_KINDS = {"introduction", "discussion", "limitations", "future_work", "conclusion",
              "threats_to_validity", "broader_impact", "results", "abstract"}


def find_cues(sections: List[Section], max_sentence_chars: int = 600) -> List[Cue]:
    cues: List[Cue] = []
    for s in sections:
        if s.kind not in _CUE_KINDS or not s.text:
            continue
        whole = s.kind in {"limitations", "future_work", "threats_to_validity"}
        for start, sent in split_sentences(s.text):
            if len(sent) < 25 or len(sent) > max_sentence_chars:
                continue
            name = next((n for n, rx in CUE_PATTERNS if rx.search(sent)), None)
            if name is None and whole:
                name = f"in_{s.kind}"        # every sentence of a Limitations section is a candidate
            if name:
                cues.append(Cue(s.idx, s.kind, sent, name, start))
    return cues


_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    return _WS.sub(" ", text).strip()
```

---

## 6. `parse_html.py`: arXiv HTML and ar5iv (both are LaTeXML output)

LaTeXML structure (what the parser relies on):
- document root `article.ltx_document` (or `div.ltx_page_main`); the abstract is `div.ltx_abstract`
- sections are `<section class="ltx_section" id="S5">`, `ltx_subsection` (`S5.SS2`), `ltx_subsubsection`, `ltx_appendix` (`A1`), and paragraph-level `ltx_paragraph`
- headings are `h2/h3/h4/h5 .ltx_title` as the first heading inside the section
- text is in `p.ltx_p`; math is `<math alttext="...">`; citations are `cite.ltx_cite`; footnotes are `.ltx_note`; the bibliography is `section.ltx_bibliography`

```python
"""LaTeXML HTML (arxiv.org/html and ar5iv) -> list[Section]."""
from __future__ import annotations

from typing import Dict, List, Optional

from .model import Section
from .sections import classify_heading, clean_heading, normalize_text

_SECTION_CLASSES = ("ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_appendix", "ltx_paragraph")
_DROP_SELECTORS = ["script", "style", "nav", "header", "footer", ".ltx_bibliography", ".ltx_page_footer",
                   ".ltx_note", ".ltx_tag_note", ".ltx_role_footnote", ".ltx_authors", ".ltx_dates",
                   ".ltx_ERROR", ".ltx_TOC", "figure.ltx_figure img", ".ltx_pagination"]


def is_latexml(html: str) -> bool:
    return "ltx_document" in html or "ltx_page_main" in html


def _text(el) -> str:
    for m in el.find_all("math"):
        alt = (m.get("alttext") or "").strip()
        m.replace_with(f" {alt} " if 0 < len(alt) <= 40 else " [MATH] ")
    for c in el.find_all("cite"):
        c.replace_with(" [CIT] ")
    return normalize_text(el.get_text(" "))


def parse_html(html: str, keep_kinds: set[str], max_section_chars: int) -> List[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one("article.ltx_document") or soup.select_one("div.ltx_page_main") or soup
    for sel in _DROP_SELECTORS:
        for el in root.select(sel):
            el.decompose()

    out: List[Section] = []
    abstract = root.select_one("div.ltx_abstract")
    if abstract is not None:
        title = abstract.find(["h6", "h2", "h3"])
        if title is not None:
            title.decompose()
        out.append(Section(0, None, "Abstract", "abstract", _text(abstract)[:max_section_chars], "abstract"))

    elements = [el for el in root.find_all("section") if any(c in (el.get("class") or []) for c in _SECTION_CLASSES)]
    index_of: Dict[int, int] = {}
    for el in elements:
        head = el.find(["h2", "h3", "h4", "h5", "h6"], class_="ltx_title")
        heading = clean_heading(head.get_text(" ")) if head is not None else ""
        parent_el = el.find_parent("section")
        parent_idx: Optional[int] = index_of.get(id(parent_el)) if parent_el is not None else None
        parent_kind = out[parent_idx].kind if parent_idx is not None else None
        kind = classify_heading(heading, parent_kind)
        # Paragraphs that belong to this section directly (not to a nested subsection).
        paras = [p for p in el.find_all("p", class_="ltx_p") if p.find_parent("section") is el]
        text = " ".join(_text(p) for p in paras) if kind in keep_kinds else ""
        idx = len(out)
        index_of[id(el)] = idx
        out.append(Section(idx, parent_idx, heading, kind, text[:max_section_chars], el.get("id") or ""))
    return out
```

> Note `_text` mutates the soup (replaces math/cite). That is fine because each element is read once.
> Paragraphs are collected per *owning* section, so a "Limitations" subsection nested inside
> "Discussion" becomes its own `limitations` record.

---

## 7. `parse_pdf.py`: PDF fallback

PDF text has no structure, so headings are found from lines. Backend: `pypdf` (BSD). PDF is the last fallback after arXiv HTML and ar5iv, so it is rarely used. (`pymupdf` would add font-size-based detection but is AGPL; deferred to `LATER_NOT_NOW.md`. The `backend` argument stays so it can be added later without changing callers.)

```python
"""PDF bytes -> list[Section]. Heuristic heading detection; good enough to find
Limitations / Future Work / Conclusion in most CS papers."""
from __future__ import annotations

import io
import re
from typing import List, Tuple

from .model import Section
from .sections import classify_heading, clean_heading, normalize_text

_HEADING_LINE = re.compile(
    r"^(?:(?P<num>(?:\d{1,2}|[A-Z])(?:\.\d{1,2}){0,2})\.?\s+)?(?P<title>[A-Z][A-Za-z0-9&,:\-'’ ]{2,70})$")
_KNOWN = re.compile(r"^(abstract|introduction|related work|background|preliminaries|method(s|ology)?|"
                    r"approach|experiments?|experimental (setup|results)|evaluation|results|discussion|"
                    r"limitations?|future work|conclusions?|conclusion and future work|"
                    r"limitations and future work|broader impacts?|ethics statement|threats to validity|"
                    r"acknowledg(e)?ments?|references|bibliography|appendix)\b", re.I)
_STOP = re.compile(r"^(references|bibliography)$", re.I)


def pdf_to_text(data: bytes, backend: str) -> str:
    if backend != "pypdf":
        raise ValueError(f"Unsupported PDF backend: {backend}")
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def split_pdf_text(text: str) -> List[Tuple[str, str]]:
    """[(heading, body)] from raw PDF text. Stops at References (appendix after it is kept)."""
    text = re.sub(r"-\n(?=[a-z])", "", text)            # de-hyphenate line breaks
    lines = [ln.strip() for ln in text.splitlines()]
    blocks: List[Tuple[str, List[str]]] = [("Front matter", [])]
    after_refs = False
    for ln in lines:
        if not ln:
            continue
        m = _HEADING_LINE.match(ln)
        words = ln.split()
        is_heading = bool(m) and len(words) <= 9 and (bool(m.group("num")) or bool(_KNOWN.match(m.group("title"))))
        if is_heading:
            title = m.group("title") if m else ln
            if _STOP.match(title.strip()):
                after_refs = True
                blocks.append(("References", []))
                continue
            if after_refs and not re.match(r"^(appendix|[A-Z]\s)", title, re.I) and not m.group("num"):
                continue
            blocks.append((title, []))
            continue
        blocks[-1][1].append(ln)
    return [(h, " ".join(body)) for h, body in blocks]


def parse_pdf(data: bytes, backend: str, keep_kinds: set[str], max_section_chars: int) -> List[Section]:
    out: List[Section] = []
    for heading, body in split_pdf_text(pdf_to_text(data, backend)):
        kind = classify_heading(heading, None) if heading != "Front matter" else "other"
        if heading == "Front matter":
            m = re.search(r"\babstract\b[:.\s—-]*(.+)", body, re.I | re.S)
            if m:
                out.append(Section(len(out), None, "Abstract", "abstract",
                                   normalize_text(m.group(1))[:max_section_chars]))
            continue
        text = normalize_text(body) if kind in keep_kinds else ""
        out.append(Section(len(out), None, clean_heading(heading), kind, text[:max_section_chars]))
    return out
```

---

## 8. `fetch.py`: polite fetcher with fallback order

```python
"""HTTP fetcher for arXiv full text. One shared rate limiter (>= 3 s between requests)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

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
    def __init__(self, cfg: Config, session=None):
        import requests

        f = cfg["fulltext"]
        self.cfg = cfg
        self.order: List[str] = list(f["order"])
        self.keep = set(f["keep_kinds"])
        self.max_chars = int(f["max_section_chars"])
        self.timeout = float(f["timeout_s"])
        self.retries = int(f["max_retries"])
        self.max_bytes = int(float(f["max_download_mb"]) * 1024 * 1024)
        self.limiter = RateLimiter(float(f["request_interval_s"]))
        mailto = os.environ.get("OPENALEX_MAILTO") or cfg["enrich"].get("mailto") or "n/a"
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": f"ScholarGrid/2.1 (research-starter; mailto:{mailto})"})

    def _get(self, url: str):
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            self.limiter.acquire()
            try:
                resp = self.session.get(url, timeout=self.timeout, allow_redirects=True, stream=True)
                if resp.status_code in (429, 503):
                    wait = float(resp.headers.get("Retry-After") or 10 * (attempt + 1))
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
            return FetchResult("ok", source, code, sections)
        return FetchResult("error" if errors else "unavailable", "", None, [], "; ".join(errors)[:500])
```

> `RateLimiter` is the existing class in `scholargrid/sources/__init__.py`; its `acquire()` sleeps as
> needed. Keep **one** `Fetcher` per run so all requests share it.

---

## 9. `synthetic.py`: offline full text with **planted** gap structure

The synthetic corpus (`data_ingest._synthetic`) builds titles and abstracts from per-topic term lists. Mirror that here and plant patterns that plans 03 and 06 must find:
- **recurring stated problems**: a small pool per topic, so the same problem appears in ≥ 3 papers,
- **missing combinations**: one method term is never paired with one task term,
- **narrow evaluation**: one method is always evaluated on the same single dataset.

```python
"""Deterministic synthetic full text (CI / demo). Not real science - planted structure only."""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from .model import Section

DATASETS = {"cs.CL": ["SynthGLUE", "SynthQA", "SynthMT"], "cs.CV": ["SynthNet", "SynthCOCO", "SynthSeg"],
            "cs.LG": ["SynthTab", "SynthBench"], "cs.CR": ["SynthMalware", "SynthPriv"],
            "cs.RO": ["SynthSim", "SynthGrasp"], "cs.DC": ["SynthCluster", "SynthTrace"]}
PROBLEM_POOL = [
    "it does not handle {a} under distribution shift",
    "evaluation on {b} in low-resource settings remains an open question",
    "the approach is limited to English data and its behaviour on other languages has not been explored",
    "the computational cost of {a} prevents deployment on edge devices",
    "it remains unclear whether {a} transfers to {b}",
]
NARROW_METHOD = "sequence"          # always evaluated on a single dataset (evaluation gap)
NEVER_PAIR = ("contrastive", "object detection")   # (method, task) never co-occur (combinatorial gap)


def _rng(arxiv_id: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1(arxiv_id.encode()).hexdigest()[:8], 16))


def synthetic_sections(arxiv_id: str, title: str, abstract: str, category: str) -> List[Section]:
    rng = _rng(arxiv_id)
    words = [w.strip(",.:").lower() for w in (title + " " + abstract).split()]
    a = title.split(":")[0].strip().lower() or "the method"
    b = next((w for w in words[::-1] if len(w) > 6), "new domains")
    datasets = DATASETS.get(category, ["SynthBench"])
    used = [datasets[0]] if NARROW_METHOD in title.lower() else list(rng.choice(datasets, size=min(2, len(datasets)),
                                                                               replace=False))
    problems = rng.choice(PROBLEM_POOL, size=2, replace=False)
    lim = " ".join(f"A limitation of our approach is that {p.format(a=a, b=b)}." for p in problems[:1])
    fut = f"Future work should explore {b} for {category} research. " + \
          f"It remains an open question whether {problems[1].format(a=a, b=b)}."
    return [
        Section(0, None, "Abstract", "abstract", abstract, "abstract"),
        Section(1, None, "Introduction", "introduction", f"{abstract} We address {a}.", "S1"),
        Section(2, None, "Experiments", "experiments",
                f"We evaluate {a} on {', '.join(used)} and report accuracy and F1.", "S4"),
        Section(3, None, "Limitations", "limitations", lim, "S6"),
        Section(4, None, "Conclusion and Future Work", "future_work", fut, "S7"),
    ]
```

---

## 10. `store.py`: `FulltextStore`

```python
"""On-disk full-text store (never shipped in the app bundle).

data/fulltext/status.parquet          one row per paper: status, source, fetched_at, counts, error
data/fulltext/sections/part-*.parquet appended in flushes; latest fetch per paper wins
data/fulltext/cues/part-*.parquet     gap-cue sentences (same flush cadence)
"""
from __future__ import annotations

import glob
import os
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..config import Config
from ..utils import ensure_dir
from .model import Cue, Section

STATUS_COLUMNS = ["arxiv_id", "status", "source", "fetched_at", "http_status", "n_sections", "n_chars", "error"]


class FulltextStore:
    def __init__(self, cfg: Config):
        self.dir = ensure_dir(cfg.fulltext_dir)
        self.sections_dir = ensure_dir(os.path.join(self.dir, "sections"))
        self.cues_dir = ensure_dir(os.path.join(self.dir, "cues"))
        self.status_path = os.path.join(self.dir, "status.parquet")
        self.status = (pd.read_parquet(self.status_path) if os.path.exists(self.status_path)
                       else pd.DataFrame(columns=STATUS_COLUMNS))
        self._sec_rows: List[Dict] = []
        self._cue_rows: List[Dict] = []
        self._status_rows: List[Dict] = []

    # -- writes -------------------------------------------------------------------------
    def add(self, arxiv_id: str, status: str, source: str, sections: List[Section], cues: List[Cue],
            http_status: Optional[int] = None, error: str = "") -> None:
        now = time.time()
        self._status_rows.append({"arxiv_id": arxiv_id, "status": status, "source": source, "fetched_at": now,
                                  "http_status": http_status, "n_sections": len(sections),
                                  "n_chars": sum(len(s.text) for s in sections), "error": error})
        for s in sections:
            self._sec_rows.append({**s.to_row(arxiv_id, source), "fetched_at": now})
        for c in cues:
            self._cue_rows.append({"arxiv_id": arxiv_id, "section_idx": c.section_idx, "kind": c.kind,
                                   "sentence": c.sentence, "cue": c.cue, "char_start": c.char_start,
                                   "fetched_at": now})

    def flush(self) -> None:
        stamp = f"{int(time.time() * 1000)}"
        if self._sec_rows:
            pd.DataFrame(self._sec_rows).to_parquet(os.path.join(self.sections_dir, f"part-{stamp}.parquet"),
                                                    index=False)
        if self._cue_rows:
            pd.DataFrame(self._cue_rows).to_parquet(os.path.join(self.cues_dir, f"part-{stamp}.parquet"),
                                                    index=False)
        if self._status_rows:
            new = pd.DataFrame(self._status_rows)
            self.status = (pd.concat([self.status, new], ignore_index=True)
                           .sort_values("fetched_at").drop_duplicates("arxiv_id", keep="last"))
            tmp = self.status_path + ".tmp"
            self.status.to_parquet(tmp, index=False)
            os.replace(tmp, self.status_path)          # atomic on POSIX and Windows
        self._sec_rows, self._cue_rows, self._status_rows = [], [], []

    # -- reads --------------------------------------------------------------------------
    def needs_fetch(self, ids: Iterable[str], refetch_failed_days: int) -> List[str]:
        st = self.status.set_index("arxiv_id") if len(self.status) else pd.DataFrame()
        cutoff = time.time() - refetch_failed_days * 86400
        out = []
        for aid in ids:
            if aid not in st.index:
                out.append(aid)
                continue
            row = st.loc[aid]
            if row["status"] in ("error", "unavailable") and float(row["fetched_at"]) < cutoff:
                out.append(aid)
        return out

    def _read_parts(self, directory: str, ids: Optional[set] = None) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(directory, "part-*.parquet")))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if ids is not None:
            df = df[df["arxiv_id"].isin(ids)]
        latest = df.groupby("arxiv_id")["fetched_at"].transform("max")
        return df[df["fetched_at"] == latest].reset_index(drop=True)

    def sections(self, ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return self._read_parts(self.sections_dir, set(ids) if ids is not None else None)

    def cues(self, ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return self._read_parts(self.cues_dir, set(ids) if ids is not None else None)

    def compact(self) -> None:
        """Merge part files (latest fetch per paper) into one part each."""
        for d in (self.sections_dir, self.cues_dir):
            df = self._read_parts(d)
            files = glob.glob(os.path.join(d, "part-*.parquet"))
            if df.empty:
                continue
            path = os.path.join(d, f"part-{int(time.time() * 1000)}-compact.parquet")
            df.to_parquet(path, index=False)
            for f in files:
                os.remove(f)
```

---

## 11. `run.py` and the pipeline stage

```python
"""run_fulltext: fetch/parse in-scope papers incrementally; returns stats for meta.json."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import pandas as pd

from ..config import Config
from ..scope import select_scope
from ..utils import get_logger
from .fetch import Fetcher
from .sections import find_cues
from .store import FulltextStore
from .synthetic import synthetic_sections

log = get_logger("fulltext")


def run_fulltext(df: pd.DataFrame, cfg: Config,
                 progress: Optional[Callable[[int, int], None]] = None) -> Dict:
    f = cfg["fulltext"]
    stats = {"enabled": bool(f["enabled"]), "in_scope": 0, "fetched_now": 0, "ok": 0,
             "unavailable": 0, "errors": 0, "by_source": {}}
    if not f["enabled"] or f["source"] == "none":
        return stats
    scope = select_scope(df, cfg)
    stats["in_scope"] = int(len(scope))
    store = FulltextStore(cfg)
    synthetic = f["source"] == "synthetic" or cfg["data"]["source"] == "synthetic"
    todo = store.needs_fetch(scope["arxiv_id"].astype(str), int(f["refetch_failed_days"]))
    if f.get("max_papers_per_run"):
        todo = todo[: int(f["max_papers_per_run"])]
    rows = scope.set_index(scope["arxiv_id"].astype(str))
    fetcher = None if synthetic else Fetcher(cfg)
    log.info("Full text: %d in scope, %d to fetch (%s).", len(scope), len(todo),
             "synthetic" if synthetic else "arXiv")
    try:
        for n, aid in enumerate(todo, start=1):
            r = rows.loc[aid]
            if synthetic:
                secs = synthetic_sections(aid, str(r["title"]), str(r["abstract"]), str(r["primary_category"]))
                store.add(aid, "ok", "synthetic", secs, find_cues(secs))
            else:
                assert fetcher is not None
                res = fetcher.fetch(aid)
                store.add(aid, res.status, res.source, res.sections, find_cues(res.sections),
                          res.http_status, res.error)
            stats["fetched_now"] += 1
            if n % int(f["flush_every"]) == 0:
                store.flush()
                if progress:
                    progress(n, len(todo))
    finally:
        store.flush()        # Ctrl-C keeps everything fetched so far
    st = store.status[store.status["arxiv_id"].isin(set(scope["arxiv_id"].astype(str)))]
    stats["ok"] = int((st["status"] == "ok").sum())
    stats["unavailable"] = int((st["status"] == "unavailable").sum())
    stats["errors"] = int((st["status"] == "error").sum())
    stats["by_source"] = {str(k): int(v) for k, v in st.loc[st["status"] == "ok", "source"].value_counts().items()}
    stats["coverage"] = round(stats["ok"] / max(1, stats["in_scope"]), 4)
    return stats
```

### 11.1 `scholargrid/pipeline.py`

```python
STAGES = ["ingest", "enrich", "fulltext", "embed", "analyze", "validate", "publish"]
# (plan 03 inserts "extract" after "fulltext"; plans 04/06 add "foundations"/"gaps" after "analyze")

_PROGRESS["fulltext"] = (0.18, "Reading full texts (Limitations, Future Work)…")


def stage_fulltext(ctx: Context) -> None:
    from .fulltext import run_fulltext

    with stage_timer("fulltext", ctx.timings) as info:
        stats = run_fulltext(ctx.frame("enriched"), ctx.cfg)
        ctx.save_json("fulltext_stats", stats)
        info.update(in_scope=stats["in_scope"], ok=stats["ok"], fetched_now=stats["fetched_now"])

_FUNCS["fulltext"] = stage_fulltext
```

In `stage_analyze` → `meta`, add `"fulltext": ctx.json("fulltext_stats", {})`.

Also make the `--status-file` progress hook show full-text progress, mirroring `harvest_progress`. Pass a `progress=lambda done, total: write_status(...)` when `args.status_file` is set. Do this by giving `stage_fulltext` access to `ctx.data.get("fulltext_progress")`, which `main()` sets.

---

## 12. CLI: `python -m scholargrid.fulltext`

```text
python -m scholargrid.fulltext stats                    # status counts by source / kind coverage
python -m scholargrid.fulltext show 2401.01234          # sections (kind, heading, chars) + cue sentences
python -m scholargrid.fulltext fetch 2401.01234 ...     # fetch now (debug), respects rate limit
python -m scholargrid.fulltext compact                  # merge part files
```

`stats` prints: total, ok/unavailable/error, ok by source, and the share of ok papers that have a `limitations` / `future_work` / `conclusion` section. This is the key coverage number.

---

## 13. Tests

### 13.1 Fixtures
- `tests/data/latexml_sample.html`: hand-written minimal LaTeXML with an `ltx_abstract`; `section.ltx_section#S1` "1 Introduction"; `#S5` "5 Discussion" containing `section.ltx_subsection#S5.SS1` "5.1 Limitations"; `#S6` "6 Conclusion and Future Work"; a `section.ltx_bibliography`; a `<math alttext="x^2">`; a `<cite>`; a footnote `.ltx_note`.
- `tests/data/pdf_text_sample.txt`: plain text resembling pypdf output with numbered headings and a References section followed by "A Appendix".

### 13.2 `tests/test_fulltext.py`

```python
import os

import pytest
from conftest import ROOT, make_config

from scholargrid.fulltext.model import Section
from scholargrid.fulltext.parse_html import parse_html
from scholargrid.fulltext.parse_pdf import split_pdf_text
from scholargrid.fulltext.sections import classify_heading, find_cues, split_sentences

KEEP = {"abstract", "introduction", "discussion", "limitations", "future_work", "conclusion"}


@pytest.mark.parametrize("heading,kind", [
    ("5 Limitations", "limitations"), ("Conclusion and Future Work", "future_work"),
    ("Limitations and Future Work", "limitations"), ("7. Concluding Remarks", "conclusion"),
    ("Threats to Validity", "threats_to_validity"), ("A Appendix: Proofs", "appendix"),
    ("4.2 Experimental Setup", "experiments"), ("Related Work", "related_work"),
    ("Broader Impact Statement", "broader_impact"), ("Strange Title", "other"),
])
def test_classify(heading, kind):
    assert classify_heading(heading) == kind


def test_parse_html_nested_limitations():
    html = open(os.path.join(ROOT, "tests", "data", "latexml_sample.html"), encoding="utf-8").read()
    secs = parse_html(html, KEEP, 8000)
    kinds = [s.kind for s in secs]
    assert kinds[0] == "abstract" and "limitations" in kinds and "future_work" in kinds
    lim = next(s for s in secs if s.kind == "limitations")
    assert lim.parent_idx is not None and lim.anchor == "S5.SS1"
    assert all("Bibliography" not in s.heading for s in secs)
    assert "x^2" in " ".join(s.text for s in secs)


def test_split_sentences_abbrev():
    sents = [s for _, s in split_sentences("We use e.g. BERT. It works. Future work is open.")]
    assert sents == ["We use e.g. BERT.", "It works.", "Future work is open."]


def test_cues():
    s = [Section(0, None, "Conclusion", "conclusion",
                 "Our method does not handle long documents. It is fast. We leave multilingual data for future work.")]
    names = [c.cue for c in find_cues(s)]
    # first matching pattern wins: "future work" is caught by `future_work` before `left_for_future`
    assert names == ["cannot", "future_work"]


def test_pdf_split():
    text = open(os.path.join(ROOT, "tests", "data", "pdf_text_sample.txt"), encoding="utf-8").read()
    heads = [h for h, _ in split_pdf_text(text)]
    assert "Limitations" in heads and "References" in heads


def test_fetch_fallback_order(tmp_path, monkeypatch):
    """arxiv_html 404 -> ar5iv returns non-LaTeXML (redirect) -> pdf ok."""
    from scholargrid.fulltext import fetch as F

    cfg = make_config(tmp_path)
    calls = []

    def fake_get(self, url):
        calls.append(url)
        if "arxiv.org/html" in url:
            return 404, b"", ""
        if "ar5iv" in url:
            return 200, b"<html>abs page</html>", ""
        return 200, b"%PDF", ""

    monkeypatch.setattr(F.Fetcher, "_get", fake_get)
    monkeypatch.setattr(F, "parse_pdf", lambda *a, **k: [Section(0, None, "Limitations", "limitations", "x" * 600)])
    res = F.Fetcher(cfg).fetch("2401.00001")
    assert res.status == "ok" and res.source == "pdf" and len(calls) == 3


def test_run_fulltext_synthetic_and_resume(tmp_path):
    from scholargrid.data_ingest import build_corpus
    from scholargrid.fulltext import FulltextStore, run_fulltext

    cfg = make_config(tmp_path)
    df = build_corpus(cfg)
    s1 = run_fulltext(df, cfg)
    assert s1["ok"] == len(df) and s1["fetched_now"] == len(df)
    assert run_fulltext(df, cfg)["fetched_now"] == 0          # resumable: nothing new to fetch
    secs = FulltextStore(cfg).sections()
    assert {"limitations", "future_work"} <= set(secs["kind"])
```

Note `build_corpus` may need the corpus columns `cited_by_count` for `select_scope`. `select_scope` already guards on the column existing.

### 13.3 `tests/test_pipeline.py`
Extend the existing full-run test: after `run(cfg)`, assert `meta["fulltext"]["ok"] > 0`.

---

## 14. Acceptance checks

```bash
ruff check . && mypy && pytest -q
python -m scholargrid.pipeline --config configs/ci.yaml --no-publish          # synthetic full text
python -m scholargrid.fulltext stats --config configs/ci.yaml
# Real but laptop-light (network, ~1-2 minutes): ~200 abstracts, 20 papers get full text
python -m scholargrid.pipeline --config configs/domain_nlp_tiny.yaml --no-publish
python -m scholargrid.fulltext stats --config configs/domain_nlp_tiny.yaml
```

Larger check (on GitHub, not the laptop): add a step `python -m scholargrid.fulltext stats` to the `dev-build`
workflow (plan 01 §8.2) and run it with `configs/domain_nlp_dev.yaml` and `from_stage` empty; read the numbers
from the job log.

Expected on real cs.CL papers from 2024 onward: ≥ 85% `ok`, mostly via `arxiv_html`. ≥ 40% of ok papers have a `limitations` or `future_work` section (ACL-style venues require a Limitations section, so cs.CL is high). Record the actual numbers in `Docs/runbook.md`.

## 15. Pitfalls

- **Never** parallelise arXiv requests or lower `request_interval_s` below 3. The schema enforces `ge=3.0`.
- arXiv HTML may return HTTP 200 with a "conversion failed" page. `is_latexml` guards that. ar5iv redirects unconverted papers to the abs page, which is also guarded.
- Don't store raw HTML/PDF. Parsing is deterministic, and refetching is cheaper than many GB of disk.
- Windows: `os.replace` for atomic status writes; don't hold Parquet files open across flushes.
- `clean_heading` strips a single leading capital letter used as appendix numbering ("A Appendix").
  A heading such as "A Survey of X" therefore becomes "Survey of X". That is harmless for kind
  classification, but don't use `clean_heading` output as a display title.
- PDF heading detection is heuristic. The `stats` coverage number tells you whether it is good enough. If PDF-sourced papers rarely show `limitations`, that is expected (many papers have no such section), not necessarily a bug.
