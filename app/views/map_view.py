"""Map — the interactive 2D research landscape."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from scholargrid.config import load_config
from scholargrid.search import search as semantic_search


def render() -> None:
    bundle = ui.get_bundle()
    df, labels, clusters = bundle.df, bundle.labels, bundle.clusters_meta
    focus = st.session_state.get("focus_cluster")

    ui.section_title("Explore", "Research map",
                     "Each dot is a paper. Click or lasso to inspect; use the "
                     "filters to narrow the view.")

    # --- filter bar -------------------------------------------------------
    f1, f2, f3 = st.columns([2, 2, 1.4], vertical_alignment="bottom")
    cats = sorted(df["primary_category"].dropna().unique().tolist())
    sel_cats = f1.multiselect("Categories", cats, default=[],
                              placeholder="All categories")
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    if dmin == dmax:
        date_range = (dmin, dmax)
        f2.caption(f"All papers dated {dmin}")
    else:
        date_range = f2.slider("Date range", dmin, dmax, (dmin, dmax))
    show_leads = f3.toggle("Show leads", value=False)

    mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
    if sel_cats:
        mask &= df["primary_category"].isin(sel_cats)
    view_idx = np.where(mask.to_numpy())[0]

    # --- optional search overlay (from Home "View on map") ----------------
    search_res = None
    query = st.session_state.get("q", "").strip()
    if st.session_state.get("highlight_matches") and query:
        top_k = int(load_config()["search"]["top_k"])
        search_res = semantic_search(query, bundle.embedder, bundle.embeddings,
                                     df, labels, clusters, top_k=top_k)
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(ui.keyword_pills([f"matches for “{query}”"]),
                    unsafe_allow_html=True)
        if c2.button("Clear matches", use_container_width=True):
            st.session_state["highlight_matches"] = False
            st.rerun()

    # --- layout: map + detail --------------------------------------------
    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = _landscape(df, labels, clusters, view_idx, search_res,
                         bundle.sparse_leads if show_leads else None,
                         None if search_res else focus)
        event = st.plotly_chart(fig, use_container_width=True, key="map",
                                on_select="rerun",
                                selection_mode=("points", "box", "lasso"))
    sel_idx = _selected_indices(event)

    with right:
        if search_res:
            _panel_results(df, search_res)
        elif sel_idx:
            _panel_selected(df, labels, clusters, sel_idx)
        elif focus is not None:
            _panel_focus(clusters, focus)
        else:
            _panel_hint()


# ---------------------------------------------------------------------------
def _landscape(df, labels, clusters, view_idx, search_res, leads, focus):
    fig = go.Figure()
    uniq = sorted({int(c) for c in labels[view_idx]})
    ci = 0
    for cid in uniq:
        pts = view_idx[labels[view_idx] == cid]
        if cid == -1:
            color, opacity, show = ui.NOISE_COLOR, 0.25, False
        else:
            color = ui.PALETTE[ci % len(ui.PALETTE)]
            ci += 1
            dim = focus is not None and cid != focus
            opacity, show = (0.12 if dim else 0.8), True
        fig.add_trace(go.Scattergl(
            x=df.iloc[pts]["x2d"], y=df.iloc[pts]["y2d"], mode="markers",
            name=(ui.cluster_name(clusters, cid)[:26] if cid != -1 else "unclustered"),
            marker=dict(size=7, color=color, opacity=opacity, line=dict(width=0)),
            customdata=pts.reshape(-1, 1),
            text=[f"{t[:90]}<br><span style='color:#888'>{c} · {d:%Y-%m}</span>"
                  for t, c, d in zip(df.iloc[pts]["title"], df.iloc[pts]["primary_category"],
                                     df.iloc[pts]["date"])],
            hoverinfo="text", showlegend=show,
        ))
    if search_res:
        sidx = search_res["result_indices"]
        fig.add_trace(go.Scattergl(
            x=df.iloc[sidx]["x2d"], y=df.iloc[sidx]["y2d"], mode="markers",
            name="matches",
            marker=dict(size=13, color="#111418", symbol="star", line=dict(width=1, color="#fff")),
            customdata=np.array(sidx).reshape(-1, 1),
            text=[t[:90] for t in df.iloc[sidx]["title"]], hoverinfo="text"))
    if leads:
        fig.add_trace(go.Scattergl(
            x=[ld["position_2d"][0] for ld in leads], y=[ld["position_2d"][1] for ld in leads],
            mode="markers", name="leads",
            marker=dict(size=16, color="rgba(0,0,0,0)", symbol="circle-open",
                        line=dict(width=2.5, color="#d1453b")),
            text=[ld["anchor_title"][:90] for ld in leads], hoverinfo="text"))
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=6, b=0), dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11),
                    itemsizing="constant"),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="#fbfcfd", paper_bgcolor="#fff", hoverlabel=dict(bgcolor="white"))
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
def _panel_hint() -> None:
    st.markdown(
        "<div class='sg-card'><div class='t'>Nothing selected yet</div>"
        "<div class='s'>Click a point, or drag a box/lasso around a cluster, "
        "to list the papers there. The map is a 2D view — treat it as a guide, "
        "not evidence by itself.</div></div>", unsafe_allow_html=True)


def _panel_results(df, res) -> None:
    st.markdown(f"### Matches · {len(res['results'])}")
    for r, idx in zip(res["results"][:15], res["result_indices"][:15]):
        snippet = ui.abstract_snippet(df.iloc[int(idx)].get("abstract"), 150)
        ui.paper_card(r["title"], r["arxiv_id"],
                      f"{r['primary_category']} · {r['date']} · similarity {r['score']:.2f}",
                      snippet=snippet)


def _panel_selected(df, labels, clusters, sel_idx) -> None:
    st.markdown(f"### Selected papers · {len(sel_idx)}")
    for i in sel_idx:
        r = df.iloc[i]
        snippet = ui.abstract_snippet(r.get("abstract"), 150)
        ui.paper_card(r["title"], r["arxiv_id"],
                      f"{r['primary_category']} · {r['date']:%Y-%m-%d} · "
                      f"{ui.cluster_name(clusters, int(labels[i]))}", snippet=snippet)


def _panel_focus(clusters, cid) -> None:
    info = clusters.get(cid) or clusters.get(str(cid)) or {}
    st.markdown(f"### {info.get('label', f'Area {cid}')}")
    st.markdown(ui.keyword_pills(info.get("keywords", [])[:8]), unsafe_allow_html=True)
    if st.button("Open area details", use_container_width=True):
        st.session_state["focus_cluster"] = cid
        ui.goto("areas")
