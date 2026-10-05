"""Map — the interactive 2D research landscape."""
from __future__ import annotations

import html
import textwrap

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from scholargrid.config import load_config
from scholargrid.search import search as semantic_search

_N_LABELS = 10


def render() -> None:
    bundle = ui.get_bundle()
    df, labels, clusters = bundle.df, bundle.labels, bundle.clusters_meta
    colors = ui.area_colors(clusters)
    focus = st.session_state.get("focus_cluster")

    ui.page_header("Explore", "Research map",
                   "Every dot is a paper. Similar papers sit close together.")

    left, right = st.columns([2.7, 1.3], gap="medium")

    with left, ui.panel("map"):
        f1, f2, f3 = st.columns([1.6, 1.6, .9], vertical_alignment="bottom")
        cats = sorted(df["primary_category"].dropna().unique().tolist())
        sel_cats = f1.multiselect("Category", cats, default=[], placeholder="All categories")
        dmin, dmax = df["date"].min().date(), df["date"].max().date()
        if dmin == dmax:
            date_range = (dmin, dmax)
            f2.caption(f"All papers from {ui.fmt_date(dmin)}")
        else:
            date_range = f2.slider("Published", dmin, dmax, (dmin, dmax), format="D MMM")
        show_leads = f3.toggle("Leads", value=False)

        mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
        if sel_cats:
            mask &= df["primary_category"].isin(sel_cats)
        view_idx = np.where(mask.to_numpy())[0]

        slot = st.empty()
        slot.markdown(ui.skeleton_block(), unsafe_allow_html=True)

        search_res = None
        query = st.session_state.get("q", "").strip()
        if st.session_state.get("highlight_matches") and query:
            top_k = int(load_config()["search"]["top_k"])
            search_res = semantic_search(query, bundle.embedder, bundle.embeddings,
                                         df, labels, clusters, top_k=top_k)

        fig = _landscape(df, labels, clusters, colors, view_idx, search_res,
                         bundle.sparse_leads if show_leads else None,
                         None if search_res else focus)
        event = slot.plotly_chart(
            fig, use_container_width=True, key="map", on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            config={"displaylogo": False,
                    "modeBarButtonsToRemove": ["toImage", "autoScale2d"]})
        st.caption(f"{len(view_idx):,} papers shown · drag to pan, scroll to zoom, "
                   "use box or lasso to select · double-click to reset")

    sel_idx = _selected_indices(event)

    with right, ui.panel("side"):
        if search_res:
            _side_matches(df, search_res, query)
        elif sel_idx:
            _side_selected(df, labels, clusters, sel_idx)
        elif focus is not None:
            _side_focus(bundle, focus, colors)
        else:
            _side_areas(clusters, colors)


# ---------------------------------------------------------------------------
def _hover_text(title: str, area: str, cat: str, date) -> str:
    t = "<br>".join(textwrap.wrap(html.escape(str(title)), 58)[:3])
    return f"<b>{t}</b><br>{html.escape(area)}<br>{cat} · {ui.fmt_date(date)}"


def _landscape(df, labels, clusters, colors, view_idx, search_res, leads, focus):
    fig = go.Figure()
    present = sorted({int(c) for c in labels[view_idx]})
    for cid in present:
        pts = view_idx[labels[view_idx] == cid]
        if cid == -1:
            color, opacity, size = ui.NOISE_COLOR, 0.35, 6
        else:
            color = colors.get(cid, ui.ACCENT)
            dim = focus is not None and cid != focus
            opacity, size = (0.1 if dim else 0.85), 8
        name = ui.area_name(clusters, cid)
        sub = df.iloc[pts]
        fig.add_trace(go.Scattergl(
            x=sub["x2d"], y=sub["y2d"], mode="markers", name=name,
            marker=dict(size=size, color=color, opacity=opacity, line=dict(width=0)),
            customdata=pts.reshape(-1, 1),
            text=[_hover_text(t, name, c, d) for t, c, d in
                  zip(sub["title"], sub["primary_category"], sub["date"])],
            hovertemplate="%{text}<extra></extra>", showlegend=False))

    if search_res:
        sidx = search_res["result_indices"]
        sub = df.iloc[sidx]
        fig.add_trace(go.Scattergl(
            x=sub["x2d"], y=sub["y2d"], mode="markers", name="matches",
            marker=dict(size=15, color="#161a22", symbol="star",
                        line=dict(width=1.5, color="#fff")),
            customdata=np.array(sidx).reshape(-1, 1),
            text=[_hover_text(t, "Search match", c, d) for t, c, d in
                  zip(sub["title"], sub["primary_category"], sub["date"])],
            hovertemplate="%{text}<extra></extra>", showlegend=False))

    if leads:
        fig.add_trace(go.Scattergl(
            x=[ld["position_2d"][0] for ld in leads], y=[ld["position_2d"][1] for ld in leads],
            mode="markers", name="leads",
            marker=dict(size=20, color="rgba(0,0,0,0)", symbol="circle-open",
                        line=dict(width=3, color="#d1453b")),
            text=[f"<b>Lead</b><br>{html.escape(ld['anchor_title'][:80])}" for ld in leads],
            hovertemplate="%{text}<extra></extra>", showlegend=False))

    # Name the biggest visible areas directly on the map instead of a legend.
    sized = sorted((cid for cid in present if cid != -1),
                   key=lambda c: -int((labels[view_idx] == c).sum()))
    targets = [focus] if focus is not None and focus in present else sized[:_N_LABELS]
    for cid in targets:
        pts = view_idx[labels[view_idx] == cid]
        fig.add_annotation(
            x=float(np.median(df.iloc[pts]["x2d"])), y=float(np.median(df.iloc[pts]["y2d"])),
            text=f"<b>{html.escape(ui.area_name(clusters, cid))}</b>", showarrow=False,
            font=dict(size=14, color=ui.INK), bgcolor="rgba(255,255,255,0.88)",
            bordercolor=colors.get(cid, ui.BORDER), borderwidth=1.5, borderpad=5)

    ui.apply_chart_style(fig, height=640)
    fig.update_layout(dragmode="pan", xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="#fbfcfd")
    return fig


def _selected_indices(event):
    idx = []
    try:
        pts = event["selection"]["points"]
    except Exception:
        pts = []
    for p in pts:
        cd = p.get("customdata")
        if cd is None:
            continue
        idx.append(int(cd[0]) if isinstance(cd, (list, tuple)) else int(cd))
    return list(dict.fromkeys(idx))[:30]


# ---------------------------------------------------------------------------
def _side_areas(clusters, colors) -> None:
    ui.panel_header("Areas", f"{len(clusters)}")
    st.caption("Pick one to highlight it on the map.")
    items = sorted(clusters.values(), key=lambda c: -c.get("size", 0))
    with st.container(height=560, border=False, key="arealist"):
        for info in items:
            cid = int(info["cluster_id"])
            sw, btn = st.columns([.09, .91], vertical_alignment="center", gap="small")
            sw.markdown(f"<span class='sg-swatch' style='background:{colors[cid]}'></span>",
                        unsafe_allow_html=True)
            if btn.button(f"{ui.area_name(clusters, cid)} · {info.get('size', 0)}",
                          key=f"focus_{cid}", type="tertiary", use_container_width=True):
                st.session_state["focus_cluster"] = cid
                st.rerun()


def _side_focus(bundle, cid, colors) -> None:
    clusters = bundle.clusters_meta
    info = ui.area_info(clusters, cid)
    st.markdown(f"<span class='sg-swatch' style='background:{colors.get(cid, ui.ACCENT)}'>"
                f"</span><span class='sg-area-t'>{html.escape(ui.area_name(clusters, cid))}"
                f"</span><div class='sg-area-m'>{info.get('size', 0)} papers</div>",
                unsafe_allow_html=True)
    st.markdown(ui.keyword_pills(info.get("keywords", [])[:8]), unsafe_allow_html=True)
    if st.button("Open area", type="primary", use_container_width=True):
        st.session_state["area_open"] = cid
        ui.goto("areas")
    if st.button("Show all areas", use_container_width=True):
        st.session_state["focus_cluster"] = None
        st.rerun()


def _side_matches(df, res, query) -> None:
    ui.panel_header("Matches", f"{len(res['results'])}")
    st.caption(f"Stars on the map · “{query}”")
    if st.button("Clear matches", use_container_width=True):
        st.session_state["highlight_matches"] = False
        st.rerun()
    with st.container(height=520, border=False):
        for i, r in enumerate(res["results"][:15]):
            ui.paper_card(r["title"], r["arxiv_id"], category=r["primary_category"],
                          date=r["date"], score=r["score"], delay=i)


def _side_selected(df, labels, clusters, sel_idx) -> None:
    ui.panel_header("Selected", f"{len(sel_idx)} papers")
    st.caption("Double-click the map to clear.")
    with st.container(height=560, border=False):
        for n, i in enumerate(sel_idx):
            r = df.iloc[i]
            ui.paper_card(r["title"], r["arxiv_id"], category=r["primary_category"],
                          date=r["date"], area=ui.area_name(clusters, int(labels[i])),
                          delay=n)
