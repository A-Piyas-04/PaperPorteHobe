"""About — how it works, limitations, and dataset provenance."""
from __future__ import annotations

import streamlit as st

import ui

_STEPS = [
    ("Collect", "Recent arXiv computer-science papers, cleaned and de-duplicated."),
    ("Embed", "Each title and abstract becomes a vector ({embedding}). Your searches use the same model."),
    ("Group", "Similar papers are grouped into research areas ({reduce} → {cluster})."),
    ("Name", "Each area is named from its most distinctive keywords."),
    ("Connect", "Papers are laid out in 2D ({landscape}) and linked to their most similar neighbours."),
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
- The graph is a **picture, not proof**. Distances and links can mislead.
- **Citations lag.** Very recent papers have few or none yet, so node size uses links instead.
- **Quiet ≠ new.** A sparse spot may just mean few papers were collected.
- **More papers ≠ better research.** Activity is not quality.
- **Growth needs history.** Areas are only called growing when the evidence is strong enough;
  everything else shows counts only.
- **Similar wording ≠ compatible science.**
""")

    with ui.panel("dataset"):
        snap = meta.get("snapshot", {})
        counts = meta["counts"]
        release = _release_id(bundle)
        ui.panel_header("Dataset", f"release {release}" if release else snap.get("source", ""))
        papers = max(1, counts["papers"])
        ui.stat_tiles([
            (f"{counts['papers']:,}", "Papers"),
            (counts["clusters"], "Areas"),
            (ui.fmt_date(snap.get("earliest_included_date")), "From"),
            (ui.fmt_date(snap.get("latest_included_date")), "To"),
        ])
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        span = snap.get("span_months")
        ui.stat_tiles([
            (f"{span:g} mo" if isinstance(span, (int, float)) else "—", "Time span"),
            (ui.fmt_date(meta.get("generated_at")), "Last refresh"),
            (f"{counts.get('with_citations', 0) / papers:.0%}", "With citations"),
            (f"{counts.get('with_references', 0) / papers:.0%}", "With references"),
        ])
        sampled = snap.get("n_available") and snap.get("n_available") != snap.get("n_papers")
        st.caption(
            f"Source: {snap.get('source', '?')}"
            + (f" · profile: {snap.get('profile')}" if snap.get("profile") else "")
            + (f" · stratified sample of {snap['n_available']:,} available papers" if sampled else "")
            + f" · embedding model: {meta.get('embedding_model') or b.get('embedding', '?')}")
        with st.expander("Full run metadata"):
            st.json(meta, expanded=False)

    with ui.panel("licence"):
        ui.panel_header("Data and licences")
        st.markdown("""
- **arXiv**: paper metadata via arXiv's open interfaces. Abstracts keep each paper's own licence;
  every result links back to arXiv, and no PDFs are redistributed. *Thank you to arXiv for use of
  its open access interoperability.*
- **OpenAlex**: citation counts, references and venues (CC0).
- ScholarGrid stores no personal data about visitors.
""")


def _release_id(bundle) -> str | None:
    import os

    manifest = os.path.join(bundle.path or "", "manifest.json")
    if os.path.exists(manifest):
        from scholargrid.utils import load_json

        return load_json(manifest).get("release")
    return None
