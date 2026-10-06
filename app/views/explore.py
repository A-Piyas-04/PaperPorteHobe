"""Explore — the paper graph (list | graph | details), Connected Papers style.

The heavy lifting happens client-side in the ``explorer`` React component.
This page only builds the payload once, runs semantic search when the
component asks for it, and hands requests (area pages) back to Streamlit.
"""
from __future__ import annotations

import streamlit as st

import ui
from components.explorer import explorer, is_built
from scholargrid.config import load_config
from scholargrid.search import search as semantic_search

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


@st.cache_resource(show_spinner=False)
def _payload(version: str) -> dict:
    bundle = ui.get_bundle()
    df, clusters = bundle.df, bundle.clusters_meta
    colors = ui.area_colors(clusters)

    papers = []
    for i, r in enumerate(df.itertuples(index=False)):
        papers.append({
            "i": i,
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

    areas = [{"id": int(c["cluster_id"]), "name": ui.area_name(clusters, c["cluster_id"]),
              "color": colors[int(c["cluster_id"])], "size": int(c.get("size", 0))}
             for c in sorted(clusters.values(), key=lambda c: -c.get("size", 0))]
    cats = df["primary_category"].value_counts().head(30)
    cited_share = float((df["cited_by_count"] > 0).mean()) if len(df) else 0.0

    return {
        "version": version,
        "papers": papers,
        "edges": bundle.edges,
        "neighbors": bundle.neighbors,
        "areas": areas,
        "categories": [[str(k), int(v)] for k, v in cats.items()],
        "sizeBy": "citations" if cited_share >= 0.2 else "links",
    }


def build_highlight(query: str, req=None) -> dict:
    """Run semantic search and shape the result for the component."""
    bundle = ui.get_bundle()
    top_k = int(load_config()["search"]["top_k"])
    res = semantic_search(query, bundle.embedder, bundle.embeddings, bundle.df,
                          bundle.labels, bundle.clusters_meta, top_k=top_k)
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
    payload = _payload(bundle.meta.get("generated_at", "v1"))
    focus = st.session_state.pop("focus_cluster", None)
    select = st.session_state.pop("explore_select", None)

    event = explorer(payload, highlight=st.session_state.get("explore_hl"),
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
