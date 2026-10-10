"""Session reading list with portable exports."""
import pandas as pd
import streamlit as st

import ui
from views.home import _bibtex

_FIELDS = ["paper_id", "arxiv_id", "doi", "url", "title", "authors", "venue", "date",
           "primary_category", "cited_by_count", "abstract", "sources"]


def _record(paper_id: str) -> dict | None:
    df = ui.get_bundle().df
    rows = df.loc[df.arxiv_id.astype(str) == paper_id]
    if not rows.empty:
        row = rows.iloc[0].to_dict()
        row.update(paper_id=paper_id, url=ui.paper_url(paper_id), sources=["arxiv"])
        return row
    return ui.get_searcher(ui.bundle_version()).paper(paper_id)


def toggle(paper_id: str, record: dict | None = None) -> None:
    """Add or remove a paper (collection or found online) from the reading list."""
    saved = dict(st.session_state.get("saved_papers", {}))
    if paper_id in saved:
        del saved[paper_id]
    else:
        row = record or _record(paper_id)
        if row:
            row = {k: row.get(k, "") for k in _FIELDS}
            row["paper_id"] = paper_id
            row["date"] = str(row["date"] or "")[:10]
            saved[paper_id] = row
    st.session_state.saved_papers = saved


def render() -> None:
    ui.page_header("Your next steps", "Reading list", "Keep the papers you want to read or discuss. Export your list to take it with you.")
    st.caption("Saved for this session only. Download your list before closing or reloading the app.")
    saved = st.session_state.get("saved_papers", {})
    if not saved:
        ui.note("<b>Your reading list is empty.</b> Search for an interest, then choose Save to reading list below a useful paper.")
        if st.button("Find papers", type="primary"):
            ui.goto("home")
        return
    rows = list(saved.values())
    c1, c2 = st.columns(2)
    c1.download_button("Download BibTeX", _bibtex(rows), "scholargrid_reading_list.bib", "application/x-bibtex", use_container_width=True)
    fields = ["paper_id", "arxiv_id", "doi", "title", "authors", "date", "venue", "primary_category", "url"]
    frame = pd.DataFrame(rows).reindex(columns=fields).fillna("")
    c2.download_button("Download CSV", frame.to_csv(index=False), "scholargrid_reading_list.csv", "text/csv", use_container_width=True)
    st.caption(f"{len(rows)} saved papers")
    for i, row in enumerate(rows):
        pid = row.get("paper_id") or row.get("arxiv_id")
        ui.paper_card(row["title"], pid, url=row.get("url") or None, authors=row.get("authors") or "",
                      cites=int(row.get("cited_by_count") or 0), date=row.get("date") or None,
                      category=row.get("primary_category") or None, venue=row.get("venue") or None,
                      snippet=ui.abstract_snippet(row.get("abstract")))
        st.button("Remove from reading list", key=f"remove_{pid}_{i}", on_click=toggle, args=(pid,))
