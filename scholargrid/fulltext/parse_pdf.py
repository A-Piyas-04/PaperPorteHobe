"""PDF bytes -> list[Section]. Heuristic heading detection; good enough to find
Limitations / Future Work / Conclusion in most CS papers.

PDF is the last fallback after arXiv HTML and ar5iv, so it is rarely used. Backend: pypdf (BSD);
pymupdf would detect headings by font size but is AGPL (deferred). The ``backend`` argument stays so
it can be added without changing callers.
"""
from __future__ import annotations

import io
import re
from typing import List, Tuple

from .model import Section
from .sections import classify_heading, clean_heading, normalize_text

_HEADING_LINE = re.compile(
    r"^(?:(?P<num>(?:\d{1,2}|[A-Z])(?:\.\d{1,2}){0,2})\.?\s+)?(?P<title>[A-Z][A-Za-z0-9&,:\-'\u2019 ]{2,70})$")
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
        is_heading = (m is not None and len(words) <= 9
                      and (bool(m.group("num")) or bool(_KNOWN.match(m.group("title")))))
        if is_heading and m is not None:
            title = m.group("title")
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
        if heading == "Front matter":
            m = re.search(r"\babstract\b[:.\s\u2014-]*(.+)", body, re.I | re.S)
            if m:
                out.append(Section(len(out), None, "Abstract", "abstract",
                                   normalize_text(m.group(1))[:max_section_chars]))
            continue
        kind = classify_heading(heading, None)
        text = normalize_text(body) if kind in keep_kinds else ""
        out.append(Section(len(out), None, clean_heading(heading), kind, text[:max_section_chars]))
    return out
