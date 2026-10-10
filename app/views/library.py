"""Session reading list with portable exports."""
import pandas as pd
import streamlit as st

import ui
from views.home import _bibtex


def toggle(arxiv_id: str) -> None:
    saved = dict(st.session_state.get("saved_papers", {}))
    if arxiv_id in saved:
        del saved[arxiv_id]
    else:
        df = ui.get_bundle().df
        rows = df.loc[df.arxiv_id == arxiv_id]
        if not rows.empty:
            row = rows.iloc[0].to_dict()
            row["date"] = str(row["date"])[:10]
            saved[arxiv_id] = row
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
    fields = ["arxiv_id", "title", "authors", "date", "primary_category"]
    c2.download_button("Download CSV", pd.DataFrame(rows)[fields].to_csv(index=False), "scholargrid_reading_list.csv", "text/csv", use_container_width=True)
    st.caption(f"{len(rows)} saved papers")
    for row in rows:
        ui.paper_card(row["title"], row["arxiv_id"], date=row["date"], category=row["primary_category"], snippet=ui.abstract_snippet(row.get("abstract")))
        st.button("Remove from reading list", key=f"remove_{row['arxiv_id']}", on_click=toggle, args=(row["arxiv_id"],))
