"""Reading list: an alias of My project when projects are durable; the session list otherwise."""
import pandas as pd
import streamlit as st

import ui
from scholargrid.bibtex import to_bibtex as _bibtex


def toggle(paper_id: str, record: dict | None = None) -> None:
    """Add or remove a paper (collection or found online). Kept for existing callers."""
    ui.toggle_paper(paper_id, record)


def render() -> None:
    if ui.get_repo() is not None:
        ui.goto("project")
        ui.note("The reading list is now part of <b>My project</b>.")
        return
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
