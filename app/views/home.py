"""Home / Search — the landing page and primary entry point."""
from __future__ import annotations

import datetime as dt
import html
import math

import pandas as pd
import streamlit as st

import ui

_LIKE = "like:"
_DATE_FLOOR = dt.date(1950, 1, 1)
_EXACT_STEP = 25               # exact matches revealed per "Load more"
_RELATED_STEP = 15             # related papers revealed per "Load more"
_CACHE_SIZE = 30
_SOURCE_NAMES = {"arxiv": "arXiv", "openalex": "OpenAlex", "semantic_scholar": "Semantic Scholar"}
_SORTS = {"Relevance": "relevance", "Newest": "newest", "Most cited": "most_cited",
          "Citation velocity": "citation_velocity"}


def _set_query(text: str) -> None:
    st.session_state.q = text


def _clear_query() -> None:
    st.session_state.q = ""


def _open_in_explore(query: str, items: list) -> None:
    st.session_state["explore_hl"] = {"query": query, "items": items, "req": None}
    st.session_state["focus_cluster"] = None


def _load_more(key: tuple) -> None:
    more = dict(st.session_state.get("_more", {}))
    more[key] = more.get(key, 0) + 1
    st.session_state["_more"] = more


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
    live = ui.get_config()["search"]["live"]
    st.caption(f"Searching {len(bundle.df):,} collected papers across {len(bundle.clusters_meta)} areas"
               + (", plus arXiv, OpenAlex and Semantic Scholar live" if live else "")
               + f". Latest collected paper: {ui.fmt_date(snap.get('latest_included_date'))}.")


def _filters(bundle) -> tuple[tuple, str, float]:
    df = bundle.df
    with st.expander("Filters and sorting"):
        c1, c2, c3 = st.columns([1.3, 1.6, 1])
        lo, hi = _DATE_FLOOR, max(df["date"].max().date(), dt.date.today())
        dates = c1.date_input("Published between", value=(lo, hi), min_value=lo, max_value=hi,
                              key="f_dates")
        cats = df["primary_category"].value_counts().index.tolist()[:40]
        chosen = c2.multiselect("arXiv categories", cats, key="f_cats", placeholder="All categories")
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


def _bibtex(rows: list) -> str:
    out = []
    for r in rows:
        aid = str(r.get("arxiv_id") or "")
        doi = str(r.get("doi") or "")
        pid = str(r.get("paper_id") or aid)
        key = "".join(ch if ch.isalnum() else "_" for ch in (f"arxiv{aid}" if aid else pid))
        authors = " and ".join(a.strip() for a in str(r.get("authors") or "").split(";") if a.strip())
        fields = [("title", r.get("title", "")), ("author", authors), ("year", str(r.get("date") or "")[:4])]
        if aid:
            fields += [("eprint", aid), ("archivePrefix", "arXiv"),
                       ("primaryClass", r.get("primary_category") or "")]
        if doi:
            fields.append(("doi", doi))
        if r.get("venue") and not aid:
            fields.append(("howpublished", r["venue"]))
        url = ui.paper_url(pid, r.get("url") or None) or (f"https://doi.org/{doi}" if doi else "")
        if url:
            fields.append(("url", url))
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
        out.append(f"@misc{{{key},\n{body}\n}}")
    return "\n\n".join(out) + "\n"


# ---------------------------------------------------------------------------
def _cached(ckey):
    return st.session_state.setdefault("_search_cache", {}).get(ckey)


def _remember(ckey, res: dict) -> None:
    cache = st.session_state.setdefault("_search_cache", {})
    cache[ckey] = res
    while len(cache) > _CACHE_SIZE:
        cache.pop(next(iter(cache)))


def _status_notes(res: dict) -> list[str]:
    notes = []
    busy, slow, down = [], [], []
    for name, status in (res.get("source_status") or {}).items():
        label = _SOURCE_NAMES.get(name, name)
        if status in ("rate_limited", "cooldown"):
            busy.append(label)
        elif status == "slow":
            slow.append(label)
        elif status == "error":
            down.append(label)
    if busy:
        notes.append(f"{' and '.join(busy)} is limiting requests right now, so these results come from "
                     "the other sources.")
    if slow:
        notes.append(f"{' and '.join(slow)} took too long and was skipped this time.")
    if down:
        notes.append(f"{' and '.join(down)} could not be reached.")
    return notes


def _totals_line(res: dict) -> str:
    totals = res.get("source_totals") or {}
    parts = [f"{_SOURCE_NAMES.get(k, k)} {int(v):,}" for k, v in totals.items() if v is not None]
    return " · ".join(parts)


def _section(title: str, meta: str = "") -> None:
    extra = f"<span>{html.escape(meta)}</span>" if meta else ""
    st.markdown(f"<div class='sg-section'>{html.escape(title)}{extra}</div>", unsafe_allow_html=True)


def _card(r: dict, clusters: dict, i: int) -> None:
    cid = r.get("cluster_id")
    ui.paper_card(
        r["title"], r["paper_id"], url=r.get("url") or None, authors=r.get("authors") or "",
        cites=int(r.get("cited_by_count") or 0), category=r.get("primary_category") or None,
        venue=r.get("venue") or None, date=r.get("date") or None,
        area=ui.area_name(clusters, cid) if cid is not None else None,
        snippet=ui.abstract_snippet(r.get("abstract")), terms=r.get("matched_terms", []),
        sources=[s for s in r.get("sources", []) if s != "collection"], new=bool(r.get("is_new")),
        delay=i)


def _preview(slot, res: dict, clusters: dict) -> None:
    """Collection results while the online sources are being checked (no buttons)."""
    exact = [r for r in res["results"] if r["match"] == "exact"][:_EXACT_STEP]
    related = [r for r in res["results"] if r["match"] == "related"][:_RELATED_STEP]
    with slot.container():
        if exact:
            _section("Exact matches", "from the saved collection")
            for i, r in enumerate(exact):
                _card(r, clusters, i)
        if related:
            _section("Related papers", "from the saved collection")
            for i, r in enumerate(related):
                _card(r, clusters, i)
        st.markdown(ui.skeleton_cards(2), unsafe_allow_html=True)


def _search(bundle, query: str, filters: tuple, sort: str, boost: float, pages: int,
            slot, progress_slot) -> dict:
    from scholargrid.search import SearchFilters

    searcher = ui.get_searcher(ui.bundle_version())
    if query.startswith(_LIKE):
        return searcher.similar(query[len(_LIKE):].strip(), int(ui.get_config()["search"]["top_k"]))
    f = SearchFilters(*filters) if filters else None
    kwargs = dict(filters=f, sort=sort, citation_weight=boost)
    quick = None
    if pages == 1 and searcher.live_enabled:
        quick = searcher.run(query, live=False, **kwargs)
        _preview(slot, quick, bundle.clusters_meta)

    def progress(msg: str) -> None:
        progress_slot.markdown(f"<div class='sg-note'>{html.escape(msg)}</div>", unsafe_allow_html=True)

    try:
        return searcher.run(query, pages=pages, progress=progress, **kwargs)
    except Exception as exc:  # online lookup must never break the page
        from scholargrid.monitoring import capture_exception

        capture_exception(exc)
        res = quick or searcher.run(query, live=False, **kwargs)
        return {**res, "source_status": {"online": "error"}}
    finally:
        progress_slot.empty()


def _results(bundle, query: str) -> None:
    with ui.panel("search"):
        s, c = st.columns([5, 1], vertical_alignment="center")
        s.text_input("Search", key="q", label_visibility="collapsed", max_chars=ui.max_query_chars(),
                     placeholder="Search a research topic…")
        c.button("Clear", use_container_width=True, on_click=_clear_query)
        filters, sort, boost = _filters(bundle)

    scfg = ui.get_config()["search"]
    key = (ui.bundle_version(), query, filters, sort, boost)
    more = st.session_state.get("_more", {}).get(key, 0)
    ex_show, rel_show = _EXACT_STEP * (more + 1), _RELATED_STEP * (more + 1)
    pages = max(1, math.ceil(ex_show / int(scfg["top_k_exact"])),
                math.ceil(rel_show / int(scfg["top_k_related"])))

    with ui.panel("results"):
        head = st.empty()
        progress_slot = st.empty()
        slot = st.empty()
        res = _cached(key + (pages,))
        if res is None:
            slot.markdown(ui.skeleton_cards(4), unsafe_allow_html=True)
            res = _search(bundle, query, filters, sort, boost, pages, slot, progress_slot)
            _remember(key + (pages,), res)

        clusters = bundle.clusters_meta
        similar = res.get("mode") == "similar"
        exact = [r for r in res["results"] if r["match"] == "exact"]
        related = [r for r in res["results"] if r["match"] == "related"]
        exact_shown, related_shown = exact[:ex_show], related[:rel_show]
        shown = exact_shown + related_shown
        about = int(res.get("about_total") or 0)

        with head.container():
            h, a = st.columns([4, 1.2], vertical_alignment="center")
            title = (f"Showing {len(shown):,} of about {about:,} papers" if about > len(shown)
                     else f"{len(shown):,} papers")
            sub = _totals_line(res)
            if res.get("mode") == "lexical":
                sub = (sub + " · " if sub else "") + "keyword search"
            h.markdown(
                f"<div class='sg-ph' style='border:0;margin:0;padding:0'>"
                f"<span class='sg-ph-t'>{html.escape(title)}</span>"
                f"<span class='sg-ph-m'>{html.escape(sub)}</span></div>",
                unsafe_allow_html=True)
            items = [[int(r["bundle_index"]), round(float(r["score"]), 4)]
                     for r in shown if r.get("bundle_index") is not None]
            if items and a.button("Explore results", type="primary", use_container_width=True,
                                  on_click=_open_in_explore, args=(query, items)):
                ui.goto("explore")
            pills = [f"{ui.area_name(clusters, d['cluster_id'])} · {d['count']}"
                     for d in res["cluster_distribution"][:5]]
            if pills:
                st.markdown(ui.keyword_pills(pills), unsafe_allow_html=True)

        if not res["results"]:
            slot.empty()
            ui.note("<b>No papers matched.</b> Try fewer filters or different words.")
            for n in _status_notes(res):
                ui.note(html.escape(n))
            return

        notes = []
        if res.get("expansion"):
            notes.append("Also searched: " + ", ".join(f"“{t}”" for t in res["expansion"]) + ".")
        notes += _status_notes(res)
        if not similar and not exact:
            notes.insert(0, f"No paper contains the exact phrase “{query}”. "
                            "Showing papers with a similar meaning instead.")

        from views.library import toggle

        saved = st.session_state.get("saved_papers", {})
        with slot.container():
            if notes:
                ui.note(" ".join(html.escape(n) for n in notes))
            st.caption("Start with a title that fits your question. Read the original, explore related "
                       "work, or save it for later.")
            n = 0
            for section, rows in (("Exact matches", exact_shown),
                                  ("Similar papers" if similar else "Related papers", related_shown)):
                if not rows:
                    continue
                if section == "Exact matches":
                    meta = f"{len(rows):,} of about {max(int(res.get('n_exact_total') or 0), about):,} " \
                           "containing your phrase"
                else:
                    meta = "similar in meaning"
                _section(section, meta)
                for r in rows:
                    _card(r, clusters, n)
                    pid = r["paper_id"]
                    is_saved = pid in saved
                    b1, b2, _ = st.columns([1.4, 1, 2.6])
                    b1.button("Remove from reading list" if is_saved else "Save to reading list",
                              key=f"save_{pid}_{n}", on_click=toggle, args=(pid, r))
                    b2.button("More like this", key=f"like_{pid}_{n}", type="tertiary",
                              on_click=_set_query, args=(f"{_LIKE}{pid}",))
                    n += 1

            if res.get("has_more") or len(exact) > ex_show or len(related) > rel_show:
                st.button("Load more papers", key="load_more", use_container_width=True,
                          on_click=_load_more, args=(key,))

        export = pd.DataFrame(shown).reindex(columns=[
            "rank", "match", "paper_id", "arxiv_id", "doi", "title", "authors", "venue",
            "primary_category", "date", "cited_by_count", "url"]).fillna("")
        e1, e2, _ = st.columns([1, 1, 3])
        e1.download_button("Export CSV", export.to_csv(index=False), file_name="scholargrid_results.csv",
                           mime="text/csv", use_container_width=True)
        e2.download_button("Export BibTeX", _bibtex(export.to_dict("records")),
                           file_name="scholargrid_results.bib", mime="application/x-bibtex",
                           use_container_width=True)
