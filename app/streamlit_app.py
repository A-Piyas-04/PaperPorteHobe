"""ScholarGrid — Research Landscape Explorer (Streamlit web app).

A lightweight, user-friendly front-end over precomputed artifacts (NFR-01):
only live query embedding + nearest-neighbour retrieval happen here. Heavy
analysis is done offline by the pipeline.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholargrid.artifacts import bundle_exists, load_bundle  # noqa: E402
from scholargrid.config import load_config  # noqa: E402
from scholargrid.search import search as semantic_search  # noqa: E402

st.set_page_config(page_title="ScholarGrid", page_icon="🔭", layout="wide")

NOISE_COLOR = "#d4d7dd"
PALETTE = ["#2f6feb", "#e8710a", "#1a9e6c", "#d1453b", "#7b57d6", "#c99700",
           "#0b8aa6", "#c2457f", "#6b8e23", "#8a6d3b", "#4b6bdb", "#2aa198"]
EXAMPLES = [
    "large language model agents",
    "diffusion models for images",
    "graph neural networks",
    "federated learning privacy",
    "reinforcement learning for robotics",
    "efficient transformer inference",
]

_CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
  h1 {font-size: 1.7rem !important; margin-bottom: .1rem;}
  [data-testid="stMetricValue"] {font-size: 1.25rem;}
  .sg-paper {border:1px solid #e6e8ec; border-radius:10px; padding:.6rem .8rem;
             margin-bottom:.5rem; background:#fff;}
  .sg-paper a {text-decoration:none; font-weight:600; color:#1f2430;}
  .sg-paper a:hover {color:#2f6feb;}
  .sg-meta {color:#6b7280; font-size:.8rem; margin-top:.15rem;}
  .sg-badge {display:inline-block; padding:.05rem .45rem; border-radius:999px;
             font-size:.72rem; font-weight:600; margin-left:.3rem;}
  .sg-up {background:#e7f5ee; color:#1a7f54;}
  .sg-flat {background:#eef0f3; color:#5b6472;}
  div[data-testid="stButton"] button {border-radius:999px;}
</style>
"""


# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading research landscape…")
def get_bundle():
    cfg = load_config()
    return load_bundle(cfg)


def cluster_name(meta: dict, cid: int) -> str:
    if cid == -1:
        return "Unclustered"
    info = meta.get(cid) or meta.get(str(cid)) or {}
    return info.get("label", f"cluster {cid}")


def growth_of(growth: dict, cid: int):
    g = growth["clusters"].get(str(cid)) or growth["clusters"].get(cid)
    return g


def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


def _reset():
    st.session_state.q = ""
    st.session_state.focus_cluster = None


# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    cfg = load_config()

    if not bundle_exists(cfg):
        _onboarding(cfg)
        return

    bundle = get_bundle()
    df, embeddings, labels = bundle.df, bundle.embeddings, bundle.labels
    meta, growth, leads, clusters = bundle.meta, bundle.growth, bundle.sparse_leads, bundle.clusters_meta

    ss("q", "")
    ss("focus_cluster", None)

    # ---- Header ----------------------------------------------------------
    hl, hr = st.columns([3, 2])
    with hl:
        st.title("🔭 ScholarGrid")
        st.caption("Explore the research landscape — search a topic, see the map, inspect the papers.")
    with hr:
        m1, m2, m3 = st.columns(3)
        m1.metric("Papers", f"{meta['counts']['papers']:,}")
        m2.metric("Areas", meta["counts"]["clusters"])
        m3.metric("Updated", str(meta["snapshot"].get("latest_included_date") or "—"))

    # ---- Search + settings ----------------------------------------------
    c1, c2, c3 = st.columns([6, 1, 1])
    query = c1.text_input("Search", key="q", label_visibility="collapsed",
                          placeholder="Search a research topic or question…")
    with c2.popover("⚙️ Options", width="stretch"):
        top_k = st.slider("Results to retrieve", 5, 60, int(cfg["search"]["top_k"]), step=5)
        cats = sorted(df["primary_category"].dropna().unique().tolist())
        sel_cats = st.multiselect("Categories", cats, default=[])
        dmin, dmax = df["date"].min().date(), df["date"].max().date()
        if dmin == dmax:
            date_range = (dmin, dmax)
            st.caption(f"All papers dated {dmin}")
        else:
            date_range = st.slider("Date range", dmin, dmax, (dmin, dmax))
        show_leads = st.toggle("Show investigation leads on map", value=False)
    c3.button("Clear", width="stretch", on_click=_reset)

    # Example chips (only when nothing is being searched).
    if not query:
        st.write("")
        chip_cols = st.columns(len(EXAMPLES))
        for col, ex in zip(chip_cols, EXAMPLES):
            col.button(ex, key=f"ex_{ex}", on_click=lambda e=ex: st.session_state.update(q=e),
                       width="stretch")

    # ---- Apply filters to the visible map --------------------------------
    mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
    if sel_cats:
        mask &= df["primary_category"].isin(sel_cats)
    view_idx = np.where(mask.to_numpy())[0]

    # ---- Semantic search --------------------------------------------------
    search_res = None
    if query.strip():
        search_res = semantic_search(query, bundle.embedder, embeddings, df, labels,
                                     clusters, top_k=int(top_k))

    # ---- Layout: map (left) + context panel (right) ----------------------
    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = _landscape(df, labels, clusters, view_idx, search_res,
                         leads if show_leads else None,
                         None if search_res else st.session_state.focus_cluster)
        event = st.plotly_chart(fig, width="stretch", key="map",
                                on_select="rerun",
                                selection_mode=("points", "box", "lasso"))
        st.caption("Tip: click or lasso points to inspect papers. The map is a 2D view, not evidence by itself.")

    sel_idx = _selected_indices(event)

    with right:
        if search_res:
            _panel_search(search_res, clusters)
        elif sel_idx:
            _panel_selected(df, labels, clusters, sel_idx)
        elif st.session_state.focus_cluster is not None:
            _panel_cluster(df, clusters, growth, st.session_state.focus_cluster)
        else:
            _panel_overview(clusters, growth)

    # ---- Secondary info (collapsed, low-text) ----------------------------
    st.divider()
    with st.expander("📈 Growth across research areas"):
        _growth_table(clusters, growth)
    with st.expander("🕳️ Investigation leads (sparse neighbourhoods)"):
        _leads_section(leads)
    with st.expander("🧪 How it works"):
        _methodology(meta)
    with st.expander("⚠️ Limitations — please read"):
        _limitations()
    with st.expander("ℹ️ Dataset & reproducibility"):
        st.json(meta, expanded=False)


# ---------------------------------------------------------------------------
# Onboarding (no artifacts yet)
# ---------------------------------------------------------------------------
def _onboarding(cfg) -> None:
    st.title("🔭 ScholarGrid")
    st.caption("First run — there is no research landscape yet.")
    st.write("Choose how to get started:")
    a, b = st.columns(2)
    with a:
        st.subheader("Try it now")
        st.write("Build a small offline demo landscape (~5 seconds, no internet).")
        if st.button("Load demo data", type="primary"):
            from scholargrid.runner import run

            demo = load_config()
            demo.raw["data"]["source"] = "synthetic"
            demo.raw["data"]["max_papers"] = 800
            bar = st.progress(0.0, "Starting…")
            run(demo, progress=lambda f, m: bar.progress(min(f, 1.0), m))
            st.cache_resource.clear()
            st.success("Demo landscape ready.")
            st.rerun()
    with b:
        st.subheader("Use real arXiv data")
        st.write("Harvest recent arXiv CS papers, then reload this page.")
        st.code("python pipeline/run_pipeline.py", language="bash")
        st.caption("Configure the window and sources in configs/config.yaml.")


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def _landscape(df, labels, clusters, view_idx, search_res, leads, focus):
    fig = go.Figure()
    uniq = sorted({int(c) for c in labels[view_idx]})
    ci = 0
    for cid in uniq:
        pts = view_idx[labels[view_idx] == cid]
        if cid == -1:
            color, opacity, show = NOISE_COLOR, 0.25, False
        else:
            color = PALETTE[ci % len(PALETTE)]
            ci += 1
            dim = focus is not None and cid != focus
            opacity, show = (0.12 if dim else 0.75), True
        fig.add_trace(go.Scattergl(
            x=df.iloc[pts]["x2d"], y=df.iloc[pts]["y2d"], mode="markers",
            name=(cluster_name(clusters, cid)[:26] if cid != -1 else "unclustered"),
            marker=dict(size=6, color=color, opacity=opacity, line=dict(width=0)),
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
            marker=dict(size=12, color="#111418", symbol="star", line=dict(width=1, color="#fff")),
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
        height=600, margin=dict(l=0, r=0, t=6, b=0), dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10),
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
# Context panels
# ---------------------------------------------------------------------------
def _paper_card(title, arxiv_id, sub):
    st.markdown(
        f"<div class='sg-paper'><a href='https://arxiv.org/abs/{arxiv_id}' target='_blank'>{title}</a>"
        f"<div class='sg-meta'>{sub}</div></div>", unsafe_allow_html=True)


def _growth_badge(rel):
    if rel is None:
        return ""
    cls, txt = ("sg-up", "faster than CS") if rel >= 1.0 else ("sg-flat", "slower than CS")
    return f"<span class='sg-badge {cls}'>{rel:g}× · {txt}</span>"


def _panel_search(res, clusters):
    st.subheader(f"Results · “{res['query']}”")
    if res["ambiguous"]:
        st.info("This topic spans several areas.", icon="🔀")
    chips = " ".join(
        f"<span class='sg-badge sg-flat'>{c['count']} · {c['label'] or 'unclustered'}</span>"
        for c in res["cluster_distribution"][:4])
    st.markdown(chips, unsafe_allow_html=True)
    st.write("")
    for r in res["results"][:12]:
        _paper_card(r["title"], r["arxiv_id"],
                    f"{r['primary_category']} · {r['date']} · similarity {r['score']}")


def _panel_selected(df, labels, clusters, sel_idx):
    st.subheader(f"Selected papers · {len(sel_idx)}")
    st.caption("From your click/lasso on the map.")
    for i in sel_idx:
        r = df.iloc[i]
        _paper_card(r["title"], r["arxiv_id"],
                    f"{r['primary_category']} · {r['date']:%Y-%m-%d} · {cluster_name(clusters, int(labels[i]))}")


def _panel_cluster(df, clusters, growth, cid):
    info = clusters.get(cid) or clusters.get(str(cid)) or {}
    g = growth_of(growth, cid)
    rel = g["windows"].get(str(growth["default_window"]), {}).get("relative_growth") if g else None
    st.subheader(info.get("label", f"Area {cid}"))
    st.markdown(f"{info.get('size', 0)} papers {_growth_badge(rel)}", unsafe_allow_html=True)
    st.markdown("**Keywords** · " + ", ".join(info.get("keywords", [])[:8]))
    if g and g.get("monthly_counts"):
        s = pd.Series(g["monthly_counts"]).sort_index()
        st.bar_chart(s, height=140)
    st.markdown("**Representative papers**")
    for p in info.get("representative_papers", []):
        _paper_card(p["title"], p["arxiv_id"], f"{p['primary_category']} · {p['date']}")


def _panel_overview(clusters, growth):
    st.subheader("Research areas")
    st.caption("Pick an area to explore, or search above.")
    items = sorted(clusters.values(), key=lambda c: -c.get("size", 0))
    for info in items:
        cid = info["cluster_id"]
        g = growth_of(growth, cid)
        rel = g["windows"].get(str(growth["default_window"]), {}).get("relative_growth") if g else None
        col1, col2 = st.columns([5, 2])
        if col1.button(f"{info['label']}", key=f"ov_{cid}", width="stretch"):
            st.session_state.focus_cluster = cid
            st.rerun()
        col2.markdown(f"<div style='padding-top:.45rem'>{info['size']} · {_growth_badge(rel)}</div>",
                      unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Expanders
# ---------------------------------------------------------------------------
def _growth_table(clusters, growth):
    st.caption(f"Reference {growth['reference_date']}. Relative growth >1 = faster than the "
               "whole CS corpus. Absolute recent counts shown too.")
    rows = []
    for cid, g in growth["clusters"].items():
        cid = int(cid)
        info = clusters.get(cid) or clusters.get(str(cid)) or {}
        dw = str(growth["default_window"])
        rows.append({
            "Area": info.get("label", f"cluster {cid}"),
            "Relative growth": g["windows"][dw]["relative_growth"],
            "Recent papers": g["windows"][dw]["recent_count"],
            "Stable": "yes" if g["stability"]["stable"] else "varies",
        })
    table = pd.DataFrame(rows).sort_values("Relative growth", ascending=False)
    st.dataframe(table, width="stretch", hide_index=True)


def _leads_section(leads):
    st.caption("Low-density regions near active areas, verified in high dimensions and across "
               "projections. These are **investigation leads, not proven research gaps.**")
    if not leads:
        st.write("No robust leads for the current landscape.")
        return
    for i, ld in enumerate(leads, 1):
        with st.container():
            st.markdown(f"**Lead {i}** · near *{ld.get('nearest_cluster_label') or 'unlabeled'}* "
                        f"· robustness {ld['robustness']}")
            for e in ld["evidence_papers"][:3]:
                _paper_card(e["title"], e["arxiv_id"], f"{e['primary_category']} · {e['date']}")


def _methodology(meta):
    b = meta.get("backends", {})
    st.markdown(f"""
1. **Collect** recent arXiv CS papers and clean them.
2. **Embed** title + abstract (`{b.get('embedding')}`) — same method for your searches.
3. **Reduce & cluster** (`{b.get('reduce')}` → `{b.get('cluster')}`) into research areas.
4. **Label** each area with its distinctive keywords and representative papers.
5. **Project** everything to the 2D map (`{b.get('landscape')}`).
6. **Measure growth** vs. the whole CS corpus and **find sparse leads**.
7. **Validate** search, clusters, growth and robustness (see `reports/`).
""")


def _limitations():
    st.markdown("""
- The 2D map is a **picture, not proof** — distances can mislead.
- **Sparse ≠ novel.** A gap on the map may just mean few papers were collected.
- **More papers ≠ better research.** Growth is activity, not quality.
- **Close in embedding space ≠ scientifically compatible.**
- Clusters depend on model and parameter choices.

ScholarGrid helps you *find and inspect* areas worth a deeper look. It does not
decide what is valuable, novel, or publishable.
""")


if __name__ == "__main__":
    main()
