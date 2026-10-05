"""Home / Search — the landing page and primary entry point."""
from __future__ import annotations

import streamlit as st

import ui
from scholargrid.config import load_config
from scholargrid.search import search as semantic_search


def render() -> None:
    bundle = ui.get_bundle()
    df, meta = bundle.df, bundle.meta
    ui.ss("q", "")

    query = st.session_state.get("q", "").strip()

    # --- hero / search ----------------------------------------------------
    if not query:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sg-eyebrow'>Research landscape explorer</div>",
                    unsafe_allow_html=True)
        st.markdown("# Find the papers that matter.")
        st.markdown("<p class='sg-lead'>Search a topic in plain language, "
                    "then explore the map of recent arXiv research.</p>",
                    unsafe_allow_html=True)

    st.text_input("Search", key="q", label_visibility="collapsed",
                  placeholder="Search a research topic or question…")

    if not query:
        _examples()
        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        ui.stat_tiles([
            (f"{meta['counts']['papers']:,}", "Papers"),
            (meta["counts"]["clusters"], "Research areas"),
            (str(meta["snapshot"].get("latest_included_date") or "—"), "Latest paper"),
        ])
        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 2])
        if c1.button("Explore the map", type="primary", use_container_width=True):
            ui.goto("map")
        if c2.button("Browse areas", use_container_width=True):
            ui.goto("areas")
        return

    # --- results ----------------------------------------------------------
    _results(bundle, query)


def _examples() -> None:
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.caption("Try one of these")
    cols = st.columns(3)
    for i, ex in enumerate(ui.EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.q = ex
            st.rerun()


def _results(bundle, query: str) -> None:
    top_k = int(load_config()["search"]["top_k"])
    res = semantic_search(query, bundle.embedder, bundle.embeddings, bundle.df,
                          bundle.labels, bundle.clusters_meta, top_k=top_k)

    head, action = st.columns([4, 1], vertical_alignment="center")
    head.markdown(f"### Results for “{query}”")
    if action.button("View on map", use_container_width=True):
        st.session_state["highlight_matches"] = True
        ui.goto("map")

    if res["ambiguous"]:
        st.info("This topic spans several research areas.", icon=":material/alt_route:")

    pills = [f"{c['count']} · {c['label'] or 'unclustered'}"
             for c in res["cluster_distribution"][:5]]
    st.markdown(ui.keyword_pills(pills), unsafe_allow_html=True)
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    df = bundle.df
    for r, idx in zip(res["results"], res["result_indices"]):
        snippet = ui.abstract_snippet(df.iloc[int(idx)].get("abstract"))
        ui.paper_card(
            r["title"], r["arxiv_id"],
            f"{r['primary_category']} · {r['date']} · "
            f"{r['cluster_label'] or 'unclustered'} · similarity {r['score']:.2f}",
            snippet=snippet,
        )
