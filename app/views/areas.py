"""Research areas — grid of clustered topics, plus a per-area detail view."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui


def _open(cid: int) -> None:
    st.session_state["area_open"] = cid


def _close() -> None:
    st.session_state["area_open"] = None


def render() -> None:
    bundle = ui.get_bundle()
    open_id = st.session_state.get("area_open")
    if open_id is not None and ui.area_info(bundle.clusters_meta, open_id):
        _detail(bundle, int(open_id))
    else:
        _grid(bundle)


# ---------------------------------------------------------------------------
def _top_cats(info: dict, n: int = 2) -> str:
    cats = list((info.get("top_categories") or {}).keys())[:n]
    return " · ".join(cats)


def _grid(bundle) -> None:
    clusters = bundle.clusters_meta
    colors = ui.area_colors(clusters)
    ui.page_header("Browse", "Research areas",
                   f"{len(clusters)} topics found in the papers, largest first.")

    items = sorted(clusters.values(), key=lambda c: -c.get("size", 0))
    delays = "".join(f".st-key-card-{int(c['cluster_id'])}{{animation-delay:{min(i, 15) * 40}ms}}"
                     for i, c in enumerate(items))
    st.markdown(f"<style>{delays}</style>", unsafe_allow_html=True)

    for start in range(0, len(items), 3):
        row = st.columns(3, gap="medium")
        for col, info in zip(row, items[start:start + 3]):
            cid = int(info["cluster_id"])
            with col, st.container(key=f"card-{cid}"):
                cats = _top_cats(info)
                st.markdown(
                    f"<span class='sg-swatch' style='background:{colors[cid]}'></span>"
                    f"<span class='sg-area-t'>{html.escape(ui.area_name(clusters, cid))}</span>"
                    f"<div class='sg-area-m'>{info.get('size', 0)} papers"
                    f"{' · ' + html.escape(cats) if cats else ''}</div>"
                    f"{ui.keyword_pills(info.get('keywords', [])[:4])}",
                    unsafe_allow_html=True)
                st.button("Explore", key=f"open_{cid}", use_container_width=True,
                          on_click=_open, args=(cid,))
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def _detail(bundle, cid: int) -> None:
    clusters, growth, df = bundle.clusters_meta, bundle.growth, bundle.df
    info = ui.area_info(clusters, cid)
    colors = ui.area_colors(clusters)
    ready = ui.growth_ready(bundle)

    st.button("← All areas", type="tertiary", on_click=_close)

    with ui.panel("area-head"):
        cats = _top_cats(info, 3)
        badge = ui.growth_badge(ui.relative_growth(growth, cid), ready)
        st.markdown(
            f"<div class='sg-eyebrow'><span class='sg-swatch' style='background:"
            f"{colors.get(cid, ui.ACCENT)}'></span>Research area</div>"
            f"<div class='sg-title'>{html.escape(ui.area_name(clusters, cid))}</div>"
            f"<p class='sg-lead' style='margin-bottom:.4rem'>{info.get('size', 0)} papers"
            f"{' · ' + html.escape(cats) if cats else ''} {badge}</p>"
            f"{ui.keyword_pills(info.get('keywords', [])[:10])}",
            unsafe_allow_html=True)
        c1, _ = st.columns([1, 2.5])
        if c1.button("See on map", type="primary", use_container_width=True):
            st.session_state["focus_cluster"] = cid
            st.session_state["highlight_matches"] = False
            ui.goto("map")

    left, right = st.columns([1.5, 1], gap="medium")
    with left, ui.panel("area-papers"):
        reps = info.get("representative_papers", [])
        ui.panel_header("Representative papers", f"{len(reps)}")
        by_id = df.set_index("arxiv_id")["abstract"] if "abstract" in df else pd.Series(dtype=str)
        for i, p in enumerate(reps):
            ui.paper_card(p["title"], p["arxiv_id"], category=p["primary_category"],
                          date=p["date"], snippet=ui.abstract_snippet(by_id.get(p["arxiv_id"])),
                          delay=i)

    with right, ui.panel("area-activity"):
        g = ui.growth_of(growth, cid)
        if ready and g and g.get("monthly_counts"):
            ui.panel_header("Publications per month")
            s = pd.Series(g["monthly_counts"]).sort_index()
            fig = go.Figure(go.Bar(x=list(s.index), y=list(s.values),
                                   marker=dict(color=colors.get(cid, ui.ACCENT))))
            ui.apply_chart_style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            ui.panel_header("Categories")
            ui.note(f"<b>Not enough history for trends.</b> "
                    f"The data covers {ui.history_days(bundle)} days.")
            cats = info.get("top_categories") or {}
            if not cats:
                cats = df.loc[bundle.labels == cid, "primary_category"].value_counts().head(5).to_dict()
            rows = [{"Category": k, "Papers": int(v)} for k, v in cats.items()]
            ui.html_table(rows, ["Category", "Papers"], numeric=("Papers",), bar_col="Papers")
