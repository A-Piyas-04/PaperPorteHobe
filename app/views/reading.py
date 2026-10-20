"""Read first — the topic's ordered reading path, with reasons, saving and exports."""
from __future__ import annotations

import streamlit as st

import ui
from scholargrid.bibtex import to_bibtex as _bibtex

_STEPS = [("survey", "1 · Get the big picture"), ("foundation", "2 · Foundations"),
          ("recent", "3 · Key recent work"), ("frontier", "4 · The frontier")]


def _markdown(ws, path: list[dict]) -> str:
    lines = [f"# Reading path: {ws.query}", ""]
    n = 1
    for step, label in _STEPS:
        items = [p for p in path if p.get("step") == step]
        if not items:
            continue
        lines += [f"## {label}", ""]
        for p in items:
            year = f" ({p['year']})" if p.get("year") else ""
            reason = p.get("why_read") or p.get("reason") or ""
            lines.append(f"{n}. [{p['title']}]({p.get('url') or ui.paper_url(str(p['paper_id']))}){year}: {reason}")
            n += 1
        lines.append("")
    return "\n".join(lines)


def _bib_rows(path: list[dict]) -> list[dict]:
    rows = []
    for p in path:
        pid = str(p.get("paper_id") or "")
        is_arxiv = bool(pid) and not pid.startswith("W") and ":" not in pid
        rows.append({"paper_id": pid, "arxiv_id": pid if is_arxiv else "", "title": p.get("title", ""),
                     "date": str(p.get("year") or ""), "url": p.get("url") or "", "authors": ""})
    return rows


def _fallback(ws) -> list[dict]:
    papers = sorted((p for p in ws.papers if p.bundle_index is not None), key=lambda p: -p.cited_by_count)[:10]
    return [{"step": "recent", "paper_id": p.paper_id, "title": p.title, "year": p.date[:4] or None,
             "url": p.url, "cited_by_count": p.cited_by_count, "in_corpus": True,
             "reason": "One of the most-cited papers retrieved for this topic."} for p in papers]


def render() -> None:
    ui.journey_bar("reading")
    ws = ui.require_workspace()
    if ws is None:
        return
    path = list(ws.reading_path)
    ui.page_header("Read first", f"What to read for: {ws.query}",
                   "An ordered path from overviews to the newest work, with a reason for each paper.")
    if not path:
        ui.note("<b>No reading path from citations yet.</b> This build has no reference data for these papers, "
                "so here are the most-cited papers retrieved for your topic instead.")
        path = _fallback(ws)
    else:
        n_citing = int(ws.stats.get("n_citing", 0))
        n_coll = int(ws.stats.get("n_collection", 0))
        st.caption(f"Foundations are based on the references of {n_citing:,} of {n_coll:,} papers in this topic "
                   "(citation data from OpenAlex).")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Export Markdown", _markdown(ws, path), "reading_path.md", "text/markdown",
                       use_container_width=True)
    c2.download_button("Export BibTeX", _bibtex(_bib_rows(path)), "reading_path.bib", "application/x-bibtex",
                       use_container_width=True)
    saved = ui.saved_paper_ids()
    n = 0
    for step, label in _STEPS:
        items = [p for p in path if p.get("step") == step]
        if not items:
            continue
        with ui.panel(f"read-{step}"):
            ui.panel_header(label, f"{len(items)}")
            for p in items:
                snippet = p.get("why_read") or p.get("reason") or ""
                if p.get("what_you_learn"):
                    snippet += f" — What you'll learn: {p['what_you_learn']}"
                pid = str(p.get("paper_id") or "")
                ui.paper_card(p.get("title", ""), pid, url=p.get("url") or None, authors="",
                              cites=int(p.get("cited_by_count") or 0), date=str(p.get("year") or ""),
                              snippet=snippet, delay=n, outside=not p.get("in_corpus", True))
                record = None if p.get("in_corpus", True) else {"paper_id": pid, "title": p.get("title", ""),
                                                                "date": str(p.get("year") or ""),
                                                                "url": p.get("url") or "",
                                                                "cited_by_count": p.get("cited_by_count", 0)}
                st.button(ui.paper_save_label(pid in saved), key=f"read_save_{n}_{pid}", on_click=ui.toggle_paper,
                          args=(pid, record, "reading_path"))
                n += 1
    st.session_state.j_read = True
