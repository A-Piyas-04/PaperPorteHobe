"""Explore — the paper graph (list | graph | details), Connected Papers style.

The heavy lifting happens client-side in the ``explorer`` React component.
This page only builds the payload once, runs semantic search when the
component asks for it, and hands requests (area pages) back to Streamlit.

Large corpora are capped at ``app.max_graph_nodes`` papers, sampled per area
(most-cited first), and re-indexed so the component's positional indices stay
valid; search highlights are mapped onto the rendered subset.
"""
from __future__ import annotations

import numpy as np
import streamlit as st
from components.explorer import explorer, is_built

import ui

_ABSTRACT_CHARS = 900
_PAGE_CSS = """
<style>
  .block-container { max-width: 100% !important; padding-left: 1.4rem !important;
                     padding-right: 1.4rem !important; padding-bottom: .5rem !important; }
  .st-key-topnav { margin-bottom: .8rem !important; }
  [data-testid="stCustomComponentV1"] { border: none; display: block; }
</style>
"""


def _truncate(text, n: int = _ABSTRACT_CHARS) -> str:
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _select_nodes(bundle, cap: int) -> np.ndarray:
    df = bundle.df
    n = len(df)
    if n <= cap:
        return np.arange(n)
    order = df.assign(_i=np.arange(n)).sort_values("cited_by_count", ascending=False)
    picks = []
    for _, g in order.groupby("cluster_id", sort=False):
        quota = max(1, int(round(cap * len(g) / n)))
        picks.append(g["_i"].to_numpy()[:quota])
    return np.sort(np.concatenate(picks)[:cap])


@st.cache_resource(show_spinner=False)
def _payload(version: str) -> tuple[dict, dict]:
    bundle = ui.get_bundle()
    df, clusters = bundle.df, bundle.clusters_meta
    colors = ui.area_colors(clusters)
    keep = _select_nodes(bundle, int(ui.get_config()["app"]["max_graph_nodes"]))
    pos = {int(orig): p for p, orig in enumerate(keep)}

    papers = []
    sub = df.iloc[keep]
    for p, r in enumerate(sub.itertuples(index=False)):
        papers.append({
            "i": p,
            "id": str(r.arxiv_id),
            "t": str(r.title),
            "au": str(r.authors or ""),
            "d": r.date.strftime("%Y-%m-%d"),
            "c": int(r.cluster_id),
            "cat": str(r.primary_category),
            "x": round(float(r.x2d), 4),
            "y": round(float(r.y2d), 4),
            "ci": int(r.cited_by_count),
            "rf": int(r.reference_count),
            "v": str(r.venue or ""),
            "ab": _truncate(r.abstract),
        })

    if len(keep) == len(df):
        edges, neighbors = bundle.edges, bundle.neighbors
    else:
        edges = [[pos[a], pos[b], w] for a, b, w in bundle.edges if a in pos and b in pos]
        neighbors = {str(pos[int(k)]): [[pos[j], s] for j, s in v if j in pos]
                     for k, v in bundle.neighbors.items() if int(k) in pos}

    areas = [{"id": int(c["cluster_id"]), "name": ui.area_name(clusters, c["cluster_id"]),
              "color": colors[int(c["cluster_id"])], "size": int(c.get("size", 0))}
             for c in sorted(clusters.values(), key=lambda c: -c.get("size", 0))]
    cats = sub["primary_category"].value_counts().head(30)
    cited_share = float((sub["cited_by_count"] > 0).mean()) if len(sub) else 0.0

    payload = {
        "version": version,
        "papers": papers,
        "edges": edges,
        "neighbors": neighbors,
        "areas": areas,
        "categories": [[str(k), int(v)] for k, v in cats.items()],
        "sizeBy": "citations" if cited_share >= 0.2 else "links",
    }
    return payload, pos


def _remap(hl: dict | None, pos: dict) -> dict | None:
    if not hl:
        return hl
    items = [[pos[int(i)], s] for i, s in hl.get("items", []) if int(i) in pos]
    return {**hl, "items": items}


def build_highlight(query: str, req=None) -> dict:
    """Run search and shape the result for the component (original indices)."""
    top_k = int(ui.get_config()["search"]["top_k"])
    res = ui.run_search(ui.bundle_version(), query[: ui.max_query_chars()], top_k)
    items = [[int(i), round(float(r["score"]), 4)]
             for i, r in zip(res["result_indices"], res["results"])]
    return {"query": query, "items": items, "req": req, "ambiguous": bool(res.get("ambiguous"))}


def render() -> None:
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)
    if not is_built():
        ui.note("The explorer component has not been built yet. Run "
                "<code>npm install</code> and <code>npm run build</code> in "
                "<code>app/components/explorer/frontend</code>, then reload.")
        return

    bundle = ui.get_bundle()
    payload, pos = _payload(ui.bundle_version())
    if len(pos) < len(bundle.df):
        st.caption(f"Showing {len(pos):,} of {len(bundle.df):,} papers (most-cited per area). "
                   "Search still covers every paper.")
    focus = st.session_state.pop("focus_cluster", None)
    select = st.session_state.pop("explore_select", None)
    if select is not None:
        select = pos.get(int(select))

    event = explorer(payload, highlight=_remap(st.session_state.get("explore_hl"), pos),
                     focus_area=focus, select=select, key="explorer")

    if not event or event.get("nonce") == st.session_state.get("_explore_nonce"):
        return
    st.session_state["_explore_nonce"] = event.get("nonce")
    kind = event.get("type")
    if kind == "search" and str(event.get("q", "")).strip():
        st.session_state["explore_hl"] = build_highlight(str(event["q"]).strip(), event.get("nonce"))
        st.rerun()
    elif kind == "clear":
        st.session_state["explore_hl"] = None
        st.rerun()
    elif kind == "open_area" and event.get("id") is not None:
        st.session_state["area_open"] = int(event["id"])
        ui.goto("areas")
