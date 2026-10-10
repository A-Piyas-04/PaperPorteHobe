"""Readable area and paper workspace, backed by a bounded React payload.

Search covers the full corpus; matches are added to the browsing sample.
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


@st.cache_resource(show_spinner=False, max_entries=4)
def _payload(version: str, required: tuple[int, ...] = ()) -> tuple[dict, dict]:
    bundle = ui.get_bundle()
    df, clusters = bundle.df, bundle.clusters_meta
    colors = ui.area_colors(clusters)
    keep = _select_nodes(bundle, int(ui.get_config()["app"]["max_graph_nodes"]))
    # Search matches must remain accessible even outside the overview sample.
    keep = np.union1d(keep, [i for i in required if 0 <= i < len(df)]).astype(int)
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

    # Rendered count and embedding centroid per area, for the overview bubbles.
    rendered_count: dict[int, int] = {}
    cx: dict[int, float] = {}
    cy: dict[int, float] = {}
    for p in papers:
        cid = p["c"]
        if cid < 0:
            continue
        rendered_count[cid] = rendered_count.get(cid, 0) + 1
        cx[cid] = cx.get(cid, 0.0) + p["x"]
        cy[cid] = cy.get(cid, 0.0) + p["y"]

    areas = []
    for c in sorted(clusters.values(), key=lambda c: -c.get("size", 0)):
        cid = int(c["cluster_id"])
        rc = rendered_count.get(cid, 0)
        area = {"id": cid, "name": ui.area_name(clusters, cid),
                "color": colors[cid], "size": int(c.get("size", 0)), "rc": rc}
        if rc:
            area["x"] = round(cx[cid] / rc, 4)
            area["y"] = round(cy[cid] / rc, 4)
        areas.append(area)

    # Aggregated inter-area links for the overview graph (summed edge weight
    # between distinct areas; similarity + bibliographic coupling, undirected).
    paper_area = [p["c"] for p in papers]
    pair_w: dict[tuple[int, int], float] = {}
    for a, b, w in edges:
        ca, cb = paper_area[a], paper_area[b]
        if ca < 0 or cb < 0 or ca == cb:
            continue
        key = (ca, cb) if ca < cb else (cb, ca)
        pair_w[key] = pair_w.get(key, 0.0) + float(w)
    area_edges = [[a, b, round(w, 3)] for (a, b), w in pair_w.items()]

    cats = sub["primary_category"].value_counts().head(30)
    cited_share = float((sub["cited_by_count"] > 0).mean()) if len(sub) else 0.0

    payload = {
        "version": version,
        "papers": papers,
        "edges": edges,
        "neighbors": neighbors,
        "areas": areas,
        "areaEdges": area_edges,
        "categories": [[str(k), int(v)] for k, v in cats.items()],
        "sizeBy": "citations" if cited_share >= 0.2 else "links",
        "corpusTotal": int(len(df)),
        "renderedTotal": int(len(keep)),
        "maxAreaPapers": int(ui.get_config()["app"].get("max_area_papers", 400)),
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
    hl = st.session_state.get("explore_hl") or {}
    required = tuple(int(i) for i, _ in hl.get("items", []))
    payload, pos = _payload(ui.bundle_version(), required)
    if len(pos) < len(bundle.df):
        st.caption(f"Showing {len(pos):,} of {len(bundle.df):,} papers (most-cited per area). "
                   "Search still covers every paper.")
    focus = st.session_state.pop("focus_cluster", None)
    select = st.session_state.pop("explore_select", None)
    if select is not None:
        select = pos.get(int(select))

    event = explorer(payload, highlight=_remap(st.session_state.get("explore_hl"), pos),
                     focus_area=focus, select=select,
                     height=960,
                     saved_ids=list(st.session_state.get("saved_papers", {})), key="explorer")

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
    elif kind == "save" and event.get("id"):
        from views.library import toggle

        toggle(str(event["id"]))
        st.rerun()
