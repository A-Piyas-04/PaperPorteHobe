"""Research areas — grid of clustered topics, plus a per-area detail view."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import ui


def render() -> None:
    bundle = ui.get_bundle()
    open_id = st.session_state.get("area_open")
    if open_id is not None:
        _detail(bundle, open_id)
    else:
        _grid(bundle)


# ---------------------------------------------------------------------------
def _grid(bundle) -> None:
    clusters, growth = bundle.clusters_meta, bundle.growth
    ui.section_title("Browse", "Research areas",
                     "Topic clusters discovered in the corpus, largest first.")

    items = sorted(clusters.values(), key=lambda c: -c.get("size", 0))
    cols = st.columns(3, gap="medium")
    for i, info in enumerate(items):
        cid = info["cluster_id"]
        rel = ui.relative_growth(growth, cid)
        with cols[i % 3]:
            st.markdown(
                f"<div class='sg-card sg-area'>"
                f"<div class='t'>{info.get('label', f'Area {cid}')}</div>"
                f"<div class='m'>{info.get('size', 0)} papers {ui.growth_badge(rel)}</div>"
                f"{ui.keyword_pills(info.get('keywords', [])[:5])}"
                f"</div>", unsafe_allow_html=True)
            if st.button("Open area", key=f"open_{cid}", use_container_width=True):
                st.session_state["area_open"] = cid
                st.rerun()


# ---------------------------------------------------------------------------
def _detail(bundle, cid) -> None:
    clusters, growth = bundle.clusters_meta, bundle.growth
    info = clusters.get(cid) or clusters.get(str(cid)) or {}
    rel = ui.relative_growth(growth, cid)

    if st.button("← All areas"):
        st.session_state["area_open"] = None
        st.rerun()

    st.markdown(f"<div class='sg-eyebrow'>Research area</div>", unsafe_allow_html=True)
    st.markdown(f"# {info.get('label', f'Area {cid}')}")
    st.markdown(f"<p class='sg-lead'>{info.get('size', 0)} papers "
                f"{ui.growth_badge(rel)}</p>", unsafe_allow_html=True)
    st.markdown(ui.keyword_pills(info.get("keywords", [])[:10]), unsafe_allow_html=True)

    if st.button("See this area on the map", type="primary"):
        st.session_state["focus_cluster"] = cid
        st.session_state["highlight_matches"] = False
        ui.goto("map")

    g = ui.growth_of(growth, cid)
    if g and g.get("monthly_counts"):
        st.markdown("### Publications over time")
        s = pd.Series(g["monthly_counts"]).sort_index()
        st.bar_chart(s, height=220, color=ui.ACCENT)

    st.markdown("### Representative papers")
    for p in info.get("representative_papers", []):
        ui.paper_card(p["title"], p["arxiv_id"], f"{p['primary_category']} · {p['date']}")
