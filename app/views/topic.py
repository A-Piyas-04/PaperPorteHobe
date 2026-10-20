"""Topic — the query-scoped workspace: overview, sub-topics in plain language, and their papers."""
from __future__ import annotations

import html

import streamlit as st

import ui
from scholargrid.workspace.activity import recently_active


def _select(sid: int | None) -> None:
    st.session_state["ws_selected"] = sid


def _show_on_map(ws, sid: int) -> None:
    idx = ws.bundle_indices(sid)
    st.session_state["explore_hl"] = {"query": ws.query, "items": [[int(i), 1.0] for i in idx], "req": None}
    st.session_state["focus_cluster"] = None
    ui.goto("explore")


def _facet_pills(s, facet: str, n: int = 4) -> str:
    vals = [v for v, _ in (s.top_facets.get(facet) or [])[:n]]
    return ui.keyword_pills(vals) if vals else ""


def _card(s) -> None:
    active = " · recently active" if recently_active(s.size, s.recent_share) else ""
    body = (f"<div class='sg-area-t'>{html.escape(s.name)}</div>"
            f"<div class='sg-subtopic-n'>{s.size} papers · {s.recent_share:.0%} from the last 12 months"
            f"{active}</div>")
    body += (f"<p style='margin:.2rem 0 .4rem'>{html.escape(s.summary)}</p>" if s.summary
             else ui.keyword_pills(s.keywords[:5]))
    methods, datasets = _facet_pills(s, "method"), _facet_pills(s, "dataset")
    if methods:
        body += f"<div class='sg-subtopic-n'>Methods</div>{methods}"
    if datasets:
        body += f"<div class='sg-subtopic-n'>Datasets</div>{datasets}"
    if s.n_open_problems:
        body += (f"<div class='sg-subtopic-n'>{s.n_open_problems} open problem"
                 f"{'s' if s.n_open_problems != 1 else ''} stated by authors</div>")
    st.markdown(body, unsafe_allow_html=True)


def _papers(ws, ids: list[str], key: str) -> None:
    by_id = {p.paper_id: p for p in ws.papers}
    saved = ui.saved_paper_ids()
    for i, pid in enumerate(ids):
        p = by_id.get(pid)
        if p is None:
            continue
        ui.paper_card(p.title, p.paper_id, url=p.url or None, authors=None if p.bundle_index is not None else "",
                      cites=p.cited_by_count, date=p.date or None, snippet=ui.abstract_snippet(p.abstract),
                      delay=i, new=p.is_new)
        record = None if p.bundle_index is not None else {"paper_id": p.paper_id, "title": p.title,
                                                          "abstract": p.abstract, "date": p.date, "url": p.url,
                                                          "cited_by_count": p.cited_by_count}
        st.button(ui.paper_save_label(pid in saved), key=f"{key}_save_{i}_{pid}", on_click=ui.toggle_paper,
                  args=(pid, record, "topic"))


def _detail(ws, s) -> None:
    with ui.panel("subtopic-detail"):
        st.button("← All sub-topics", type="tertiary", on_click=_select, args=(None,), key="ws_back")
        st.markdown(f"<div class='sg-eyebrow'>Sub-topic</div><div class='sg-title' style='font-size:1.6rem'>"
                    f"{html.escape(s.name)}</div>", unsafe_allow_html=True)
        st.caption(f"{s.size} papers · {s.n_collection} in the analysed collection · keywords: "
                   + ", ".join(s.keywords[:6]))
        if s.summary:
            st.write(s.summary)
        if s.key_concepts:
            rows = "".join(f"<dt>{html.escape(c.get('term', ''))}</dt><dd>{html.escape(c.get('meaning', ''))}</dd>"
                           for c in s.key_concepts)
            st.markdown("<div class='sg-section'>Key concepts</div>"
                        f"<dl class='sg-concepts'>{rows}</dl>", unsafe_allow_html=True)
        if s.beginner_entry:
            ui.note(f"<b>Where to begin:</b> {html.escape(s.beginner_entry)}")
        c1, c2, _ = st.columns([1, 1, 2])
        if s.n_collection and c1.button("Show on the map", key=f"ws_map_{s.id}", use_container_width=True):
            _show_on_map(ws, s.id)
        pages = st.session_state.get("_pages") or {}
        if "gaps" in pages and c2.button("See open problems here", key=f"ws_gaps_{s.id}",
                                         use_container_width=True):
            st.session_state["gaps_subtopic"] = s.id
            ui.goto("gaps")
        micros = ws.children(s.id)
        if micros:
            tabs = st.tabs([m.name for m in micros])
            for tab, m in zip(tabs, micros):
                with tab:
                    st.caption(f"{m.size} papers" + (f" · {m.summary}" if m.summary else ""))
                    _papers(ws, m.representative, key=f"m{m.id}")
        st.markdown("<div class='sg-section'>Representative papers</div>", unsafe_allow_html=True)
        _papers(ws, s.representative, key=f"s{s.id}")


def render() -> None:
    ui.journey_bar("topic")
    ws = ui.require_workspace()
    if ws is None:
        return
    top = sorted(ws.top_level(), key=lambda s: -s.size)
    n_live = int(ws.stats.get("n_live", 0))
    ui.page_header("Your topic", f"Topic: {ws.query}",
                   f"{len(ws.papers)} papers · {len(top)} sub-topics"
                   + (f" · {n_live} found online" if n_live else ""))
    for w in ws.warnings:
        ui.note(html.escape(w))

    selected = st.session_state.get("ws_selected")
    sub = ws.subtopic(int(selected)) if selected is not None else None
    if sub is not None:
        _detail(ws, sub)
    else:
        if ws.overview:
            with ui.panel("topic-overview"):
                ui.panel_header("Overview")
                st.write(ws.overview.get("summary", ""))
                if ws.overview.get("how_they_relate"):
                    st.markdown(f"**How the parts relate.** {ws.overview['how_they_relate']}")
                if ws.overview.get("prerequisites"):
                    st.markdown("<div class='sg-section'>Know these first</div>"
                                + ui.keyword_pills(ws.overview["prerequisites"]), unsafe_allow_html=True)
        if not top:
            ui.note("<b>No papers matched this topic.</b> Try a broader or differently worded description.")
        for start in range(0, len(top), 3):
            row = st.columns(3, gap="medium")
            for col, s in zip(row, top[start:start + 3]):
                with col, st.container(key=f"ws-card-{s.id}"):
                    _card(s)
                    st.button("Open", key=f"ws_open_{s.id}", use_container_width=True, on_click=_select,
                              args=(s.id,))
        st.caption("Sub-topics group the papers retrieved for your topic; activity is based on those papers, "
                   "not a growth estimate. See Trends for growth statistics.")

    unclustered = [p for p in ws.papers if p.subtopic < 0]
    if unclustered:
        with st.expander(f"Other papers found online ({len(unclustered)})"):
            _papers(ws, [p.paper_id for p in unclustered[:30]], key="online")
    st.session_state.j_topic = True
