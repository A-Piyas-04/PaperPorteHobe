"""About — how it works, limitations, and dataset provenance."""
from __future__ import annotations

import html

import streamlit as st

import ui

_STEPS = [
    ("Collect", "Recent arXiv computer-science papers, cleaned and de-duplicated."),
    ("Embed", "Each title and abstract becomes a vector ({embedding}). Your searches use the same model."),
    ("Group", "Similar papers are grouped into research areas ({reduce} → {cluster})."),
    ("Name", "Each area is named from its most distinctive keywords."),
    ("Connect", "Related papers are connected by text similarity, with shared-reference signals when available. Browse these connections one paper at a time."),
    ("Check", "Search quality, cluster quality and robustness are validated on every run."),
    ("Search", "Searches show papers containing your exact phrase (acronyms such as SATD are expanded) "
               "first, then papers with a similar meaning. Besides the collection, each search asks "
               "arXiv, OpenAlex and Semantic Scholar live; papers found that way are kept, so the "
               "collection grows with use."),
]


def render() -> None:
    bundle = ui.get_bundle()
    meta = bundle.meta
    b = meta.get("backends", {})

    ui.page_header("About", "How ScholarGrid works",
                   "A tool for exploring research, not for judging it.")

    with ui.panel("how"):
        ui.panel_header("The process", f"{len(_STEPS)} steps")
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
- Research areas are **automatic groupings, not an authoritative taxonomy**. Check the papers behind each label.
- **Citations lag.** Very recent papers have few or none yet. A zero count may also reflect missing metadata.
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

    _grow_panel(bundle)

    with ui.panel("licence"):
        ui.panel_header("Data and licences")
        st.markdown("""
- **arXiv**: paper metadata via arXiv's open interfaces. Abstracts keep each paper's own licence;
  every result links back to arXiv, and no PDFs are redistributed. *Thank you to arXiv for use of
  its open access interoperability.*
- **OpenAlex**: search results, citation counts, references and venues (CC0).
- **Semantic Scholar**: search results via the Semantic Scholar Academic Graph API.
- Abstracts and metadata from every source link back to the original record.
- ScholarGrid stores no personal data about visitors.
""")


def _start_grow() -> None:
    from scholargrid.grow import start

    start(ui.get_config())


def _grow_panel(bundle) -> None:
    from scholargrid.grow import log_tail, read_status

    cfg = ui.get_config()
    if not cfg["app"]["allow_grow"]:
        return
    status = read_status(cfg)
    state = status.get("state", "idle")
    target = int(cfg.max_papers or 0)
    with ui.panel("grow"):
        ui.panel_header("Grow the collection", f"{len(bundle.df):,} papers now")
        st.write(f"Collect up to {target:,} arXiv papers month by month across the whole date window. "
                 "This runs in the background (roughly 30–60 minutes) while search keeps working; "
                 "the larger collection is picked up automatically when it is ready. "
                 "Search already checks arXiv, OpenAlex and Semantic Scholar live, so this mainly "
                 "improves Areas, Trends and the research map.")
        if state == "running":
            st.progress(float(status.get("progress") or 0.0), text=status.get("message") or "Working…")
            st.caption(f"Started {ui.fmt_date(status.get('started_at'))}. Reload this page for an update.")
            st.button("Refresh status", key="grow_refresh")
            return
        if state == "done":
            ui.note(f"<b>Last run finished</b> {html.escape(ui.fmt_date(status.get('finished_at')))}"
                    + (f": release {html.escape(str(status['release']))}" if status.get("release") else "")
                    + (f" with {int(status['papers']):,} papers" if status.get("papers") else "") + ".")
        elif state == "failed":
            ui.note(f"<b>The last run failed.</b> {html.escape(str(status.get('message') or ''))} "
                    "Finished months are cached, so trying again continues where it stopped.")
            with st.expander("Log"):
                st.code(log_tail(cfg) or "(empty)")
        st.button("Try again" if state == "failed" else "Grow collection", type="primary",
                  key="grow_start", on_click=_start_grow)


def _release_id(bundle) -> str | None:
    import os

    manifest = os.path.join(bundle.path or "", "manifest.json")
    if os.path.exists(manifest):
        from scholargrid.utils import load_json

        return load_json(manifest).get("release")
    return None
