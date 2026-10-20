"""LaTeXML HTML (arxiv.org/html and ar5iv) -> list[Section].

``_text`` mutates the soup (replaces math/cite); each element is read once, so that is fine.
Paragraphs are collected per *owning* section, so a "Limitations" subsection nested inside
"Discussion" becomes its own ``limitations`` record.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .model import Section
from .sections import classify_heading, clean_heading, normalize_text

_SECTION_CLASSES = ("ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_appendix", "ltx_paragraph")
_DROP_SELECTORS = ["script", "style", "nav", "header", "footer", ".ltx_bibliography", ".ltx_page_footer",
                   ".ltx_note", ".ltx_tag_note", ".ltx_role_footnote", ".ltx_authors", ".ltx_dates",
                   ".ltx_ERROR", ".ltx_TOC", "figure.ltx_figure img", ".ltx_pagination"]


def is_latexml(html: str) -> bool:
    return "ltx_document" in html or "ltx_page_main" in html


def _text(el: Any) -> str:
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
        out.append(Section(idx, parent_idx, heading, kind, text[:max_section_chars], str(el.get("id") or "")))
    return out
