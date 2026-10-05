"""Home / Search — the landing page and primary entry point."""
from __future__ import annotations

import streamlit as st

import ui
from scholargrid.config import load_config
from scholargrid.search import search as semantic_search


def _set_query(text: str) -> None:
    st.session_state.q = text


def _clear_query() -> None:
    st.session_state.q = ""


def _view_on_map() -> None:
    st.session_state["highlight_matches"] = True
    st.session_state["focus_cluster"] = None


def render() -> None:
    bundle = ui.get_bundle()
    ui.ss("q", "")
    query = st.session_state.get("q", "").strip()

    if not query:
        _landing(bundle)
    else:
        _results(bundle, query)


# ---------------------------------------------------------------------------
def _landing(bundle) -> None:
    meta = bundle.meta
    with ui.panel("hero"):
        st.markdown(
            "<div class='sg-eyebrow'>Research landscape explorer</div>"
            "<div class='sg-title'>Find the papers that matter.</div>"
            "<p class='sg-lead'>Describe a topic in plain words. We'll find the closest "
            "recent arXiv papers and show where they sit in the field.</p>",
            unsafe_allow_html=True)
        st.text_input("Search", key="q", label_visibility="collapsed",
                      placeholder="e.g. efficient attention for long documents")
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.caption("Popular searches")
        cols = st.columns(3)
        for i, ex in enumerate(ui.EXAMPLES):
            cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True,
                               on_click=_set_query, args=(ex,))

    with ui.panel("glance"):
        ui.panel_header("At a glance")
        snap = meta.get("snapshot", {})
        ui.stat_tiles([
            (f"{meta['counts']['papers']:,}", "Papers"),
            (meta["counts"]["clusters"], "Research areas"),
            (len(bundle.sparse_leads), "Investigation leads"),
            (ui.fmt_date(snap.get("latest_included_date")), "Latest paper"),
        ])
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 1.4])
        if c1.button("Explore the map", type="primary", use_container_width=True):
            ui.goto("map")
        if c2.button("Browse areas", use_container_width=True):
            ui.goto("areas")


def _results(bundle, query: str) -> None:
    with ui.panel("search"):
        s, c = st.columns([5, 1], vertical_alignment="center")
        s.text_input("Search", key="q", label_visibility="collapsed",
                     placeholder="Search a research topic…")
        c.button("Clear", use_container_width=True, on_click=_clear_query)

    with ui.panel("results"):
        head = st.empty()
        slot = st.empty()
        slot.markdown(ui.skeleton_cards(4), unsafe_allow_html=True)

        top_k = int(load_config()["search"]["top_k"])
        res = semantic_search(query, bundle.embedder, bundle.embeddings, bundle.df,
                              bundle.labels, bundle.clusters_meta, top_k=top_k)
        clusters = bundle.clusters_meta
        n_areas = sum(1 for d in res["cluster_distribution"] if d["cluster_id"] != -1)

        with head.container():
            h, a = st.columns([4, 1.2], vertical_alignment="center")
            h.markdown(
                f"<div class='sg-ph' style='border:0;margin:0;padding:0'>"
                f"<span class='sg-ph-t'>{len(res['results'])} papers</span>"
                f"<span class='sg-ph-m'>across {n_areas} area{'s' if n_areas != 1 else ''}"
                f"</span></div>", unsafe_allow_html=True)
            if a.button("View on map", type="primary", use_container_width=True,
                        on_click=_view_on_map):
                ui.goto("map")
            pills = [f"{ui.area_name(clusters, d['cluster_id'])} · {d['count']}"
                     for d in res["cluster_distribution"][:5]]
            st.markdown(ui.keyword_pills(pills), unsafe_allow_html=True)

        df = bundle.df
        with slot.container():
            for i, (r, idx) in enumerate(zip(res["results"], res["result_indices"])):
                ui.paper_card(
                    r["title"], r["arxiv_id"], category=r["primary_category"],
                    date=r["date"], area=ui.area_name(clusters, r["cluster_id"]),
                    score=r["score"],
                    snippet=ui.abstract_snippet(df.iloc[int(idx)].get("abstract")),
                    delay=i)
