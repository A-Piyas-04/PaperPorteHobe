"""Home / Search — the landing page and primary entry point."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import ui

_LIKE = "like:"
_SORTS = {"Relevance": "relevance", "Newest": "newest", "Most cited": "most_cited",
          "Citation velocity": "citation_velocity"}


def _set_query(text: str) -> None:
    st.session_state.q = text


def _clear_query() -> None:
    st.session_state.q = ""


def _open_in_explore(query: str, items: list) -> None:
    st.session_state["explore_hl"] = {"query": query, "items": items, "req": None}
    st.session_state["focus_cluster"] = None


def render() -> None:
    bundle = ui.get_bundle()
    ui.ss("q", "")
    query = st.session_state.get("q", "").strip()[: ui.max_query_chars()]

    if not query:
        _landing(bundle)
    else:
        _results(bundle, query)


# ---------------------------------------------------------------------------
def _landing(bundle) -> None:
    from views import tour

    with ui.panel("hero"):
        st.markdown(
            "<div class='sg-eyebrow'>A clearer way into computer-science research</div>"
            "<h1 class='sg-title'>From a research interest<br>to your next useful read.</h1>"
            "<p class='sg-lead'>Find relevant papers, understand the areas around them, "
            "and build a reading list you can take into your next project or discussion.</p>",
            unsafe_allow_html=True)
        with st.form("start-search"):
            st.text_input("What would you like to understand?", key="q", max_chars=ui.max_query_chars(),
                          placeholder="e.g. how to make language models more efficient")
            submitted = st.form_submit_button("Find papers", type="primary", use_container_width=True)
        if submitted and st.session_state.q.strip():
            st.rerun()
        st.caption("Or start with an example")
        cols = st.columns(3)
        for i, ex in enumerate(ui.EXAMPLES[:3]):
            cols[i].button(ex, key=f"ex_{i}", use_container_width=True,
                           on_click=_set_query, args=(ex,))

    st.markdown("<div class='sg-journey'>"
                "<div><b>01 - Find your starting point</b><p>Search an interest or browse a research area.</p></div>"
                "<div><b>02 - Understand the context</b><p>Read abstracts and follow related papers.</p></div>"
                "<div><b>03 - Leave with a reading list</b><p>Save useful papers and export your references.</p></div>"
                "</div>", unsafe_allow_html=True)
    tour.render()
    with ui.panel("start-browsing"):
        ui.panel_header("Still choosing a topic?")
        st.write("Open the connected research map to see how areas relate, then follow a paper that interests you.")
        if st.button("Open the research map", use_container_width=True):
            ui.goto("explore")
    snap = bundle.meta.get("snapshot", {})
    st.caption(f"Searching {len(bundle.df):,} papers across {len(bundle.clusters_meta)} areas - "
               f"Latest included paper: {ui.fmt_date(snap.get('latest_included_date'))}. "
               "Results reflect this dataset, not all published research.")


def _filters(bundle) -> tuple[tuple, str, float]:
    df = bundle.df
    with st.expander("Filters and sorting"):
        c1, c2, c3 = st.columns([1.3, 1.6, 1])
        lo, hi = df["date"].min().date(), df["date"].max().date()
        dates = c1.date_input("Published between", value=(lo, hi), min_value=lo, max_value=hi,
                              key="f_dates")
        cats = df["primary_category"].value_counts().index.tolist()[:40]
        chosen = c2.multiselect("Categories", cats, key="f_cats", placeholder="All categories")
        min_cites = int(c3.number_input("Min. citations", min_value=0, value=0, step=1, key="f_cites"))
        s1, s2 = st.columns([1, 1.4])
        sort = _SORTS[s1.selectbox("Sort by", list(_SORTS), key="f_sort")]
        boost = s2.toggle("Favour well-cited papers (age-adjusted)", key="f_boost")
    d_from = d_to = None
    if isinstance(dates, (list, tuple)) and len(dates) == 2:
        if dates[0] != lo:
            d_from = dates[0].isoformat()
        if dates[1] != hi:
            d_to = dates[1].isoformat()
    filters = (d_from, d_to, tuple(chosen), min_cites, False)
    active = d_from or d_to or chosen or min_cites
    return (filters if active else ()), sort, (0.5 if boost else 0.0)


def _similar(bundle, arxiv_id: str, top_k: int) -> dict:
    df = bundle.df
    hits = df.index[df["arxiv_id"] == arxiv_id]
    if len(hits) == 0:
        return {"results": [], "result_indices": [], "cluster_distribution": [], "mode": "similar"}
    i = int(hits[0])
    vec = bundle.embeddings[i].astype("float32")
    idx, sims = bundle.index.search(vec[None, :], top_k + 1)
    order = [(int(j), float(s)) for j, s in zip(idx[0], sims[0]) if int(j) != i][:top_k]
    results = []
    for rank, (j, s) in enumerate(order, 1):
        r = df.iloc[j]
        results.append({"rank": rank, "arxiv_id": r["arxiv_id"], "title": r["title"],
                        "primary_category": r["primary_category"], "date": r["date"].strftime("%Y-%m-%d"),
                        "cluster_id": int(bundle.labels[j]), "score": round(s, 4),
                        "cited_by_count": int(r["cited_by_count"]), "matched_terms": []})
    from collections import Counter

    dist = Counter(r["cluster_id"] for r in results)
    return {"query": f"papers like “{df.iloc[i]['title']}”", "results": results,
            "result_indices": [j for j, _ in order], "mode": "similar", "degraded": False,
            "cluster_distribution": [{"cluster_id": c, "count": n} for c, n in dist.most_common()]}


def _bibtex(rows: list) -> str:
    out = []
    for r in rows:
        key = "arxiv" + str(r["arxiv_id"]).replace(".", "_").replace("/", "_")
        authors = " and ".join(a.strip() for a in str(r.get("authors", "")).split(";") if a.strip())
        out.append(f"@misc{{{key},\n  title = {{{r['title']}}},\n  author = {{{authors}}},\n"
                   f"  year = {{{str(r['date'])[:4]}}},\n  eprint = {{{r['arxiv_id']}}},\n"
                   f"  archivePrefix = {{arXiv}},\n  primaryClass = {{{r['primary_category']}}},\n"
                   f"  url = {{https://arxiv.org/abs/{r['arxiv_id']}}}\n}}")
    return "\n\n".join(out) + "\n"


def _results(bundle, query: str) -> None:
    with ui.panel("search"):
        s, c = st.columns([5, 1], vertical_alignment="center")
        s.text_input("Search", key="q", label_visibility="collapsed", max_chars=ui.max_query_chars(),
                     placeholder="Search a research topic…")
        c.button("Clear", use_container_width=True, on_click=_clear_query)
        filters, sort, boost = _filters(bundle)

    with ui.panel("results"):
        head = st.empty()
        slot = st.empty()
        slot.markdown(ui.skeleton_cards(4), unsafe_allow_html=True)

        top_k = int(ui.get_config()["search"]["top_k"])
        if query.startswith(_LIKE):
            res = _similar(bundle, query[len(_LIKE):].strip(), top_k)
        else:
            res = ui.run_search(ui.bundle_version(), query, top_k, filters, sort, boost)
        clusters = bundle.clusters_meta
        n_areas = sum(1 for d in res["cluster_distribution"] if d["cluster_id"] != -1)
        df = bundle.df

        with head.container():
            h, a = st.columns([4, 1.2], vertical_alignment="center")
            h.markdown(
                f"<div class='sg-ph' style='border:0;margin:0;padding:0'>"
                f"<span class='sg-ph-t'>{len(res['results'])} papers</span>"
                f"<span class='sg-ph-m'>across {n_areas} area{'s' if n_areas != 1 else ''}"
                f"{' · keyword search' if res.get('mode') == 'lexical' else ''}</span></div>",
                unsafe_allow_html=True)
            items = [[int(i), round(float(r["score"]), 4)]
                     for i, r in zip(res["result_indices"], res["results"])]
            if a.button("Explore results", type="primary", use_container_width=True,
                        on_click=_open_in_explore, args=(query, items)):
                ui.goto("explore")
            pills = [f"{ui.area_name(clusters, d['cluster_id'])} · {d['count']}"
                     for d in res["cluster_distribution"][:5]]
            st.markdown(ui.keyword_pills(pills), unsafe_allow_html=True)

        if not res["results"]:
            slot.empty()
            ui.note("<b>No papers matched.</b> Try fewer filters or different words.")
            return

        st.caption("Start with a title that fits your question. Read the original, explore related work, or save it for later.")
        from views.library import toggle

        with slot.container():
            for i, (r, idx) in enumerate(zip(res["results"], res["result_indices"])):
                ui.paper_card(
                    r["title"], r["arxiv_id"], category=r["primary_category"],
                    date=r["date"], area=ui.area_name(clusters, r["cluster_id"]),
                    score=None,
                    snippet=ui.abstract_snippet(df.iloc[int(idx)].get("abstract")),
                    terms=r.get("matched_terms", []), delay=i)
                saved = r["arxiv_id"] in st.session_state.get("saved_papers", {})
                st.button("Remove from reading list" if saved else "Save to reading list",
                          key=f"save_{r['arxiv_id']}_{i}", on_click=toggle, args=(r["arxiv_id"],))
                st.button("More like this", key=f"like_{r['arxiv_id']}_{i}", type="tertiary",
                          on_click=_set_query, args=(f"{_LIKE}{r['arxiv_id']}",))

        export = pd.DataFrame([{**r, "authors": df.iloc[int(idx)]["authors"]}
                               for r, idx in zip(res["results"], res["result_indices"])])
        buf = io.StringIO()
        export[["rank", "arxiv_id", "title", "authors", "primary_category", "date",
                "cited_by_count", "score"]].to_csv(buf, index=False)
        e1, e2, _ = st.columns([1, 1, 3])
        e1.download_button("Export CSV", buf.getvalue(), file_name="scholargrid_results.csv",
                           mime="text/csv", use_container_width=True)
        e2.download_button("Export BibTeX", _bibtex(export.to_dict("records")),
                           file_name="scholargrid_results.bib", mime="application/x-bibtex",
                           use_container_width=True)
