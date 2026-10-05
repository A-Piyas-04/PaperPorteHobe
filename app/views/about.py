"""About — how it works, limitations, and dataset provenance."""
from __future__ import annotations

import streamlit as st

import ui

_STEPS = [
    ("Collect", "Recent arXiv computer-science papers, cleaned and de-duplicated."),
    ("Embed", "Each title and abstract becomes a vector ({embedding}). Your searches use the same model."),
    ("Group", "Similar papers are grouped into research areas ({reduce} → {cluster})."),
    ("Name", "Each area is named from its most distinctive keywords."),
    ("Map", "Everything is laid out in 2D ({landscape}) so you can see the field."),
    ("Check", "Search quality, cluster quality and robustness are validated on every run."),
]


def render() -> None:
    bundle = ui.get_bundle()
    meta = bundle.meta
    b = meta.get("backends", {})

    ui.page_header("About", "How ScholarGrid works",
                   "A tool for exploring research, not for judging it.")

    with ui.panel("how"):
        ui.panel_header("The process", "6 steps")
        cols = st.columns(3, gap="medium")
        for i, (title, text) in enumerate(_STEPS):
            cols[i % 3].markdown(
                f"<div class='sg-card' style='animation-delay:{i * 50}ms'>"
                f"<div class='sg-eyebrow'>Step {i + 1}</div><div class='t'>{title}</div>"
                f"<div class='s'>{text.format(**{k: b.get(k, '?') for k in ('embedding', 'reduce', 'cluster', 'landscape')})}</div>"
                f"</div>", unsafe_allow_html=True)

    with ui.panel("limits"):
        ui.panel_header("Keep in mind")
        st.markdown("""
- The map is a **picture, not proof**. Distances on it can mislead.
- **Quiet ≠ new.** A sparse spot may just mean few papers were collected.
- **More papers ≠ better research.** Activity is not quality.
- **Similar wording ≠ compatible science.**
""")

    with ui.panel("dataset"):
        snap = meta.get("snapshot", {})
        ui.panel_header("Dataset", snap.get("source", ""))
        ui.stat_tiles([
            (f"{meta['counts']['papers']:,}", "Papers"),
            (meta["counts"]["clusters"], "Areas"),
            (ui.fmt_date(snap.get("earliest_included_date")), "From"),
            (ui.fmt_date(snap.get("latest_included_date")), "To"),
        ])
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        with st.expander("Full run metadata"):
            st.json(meta, expanded=False)
