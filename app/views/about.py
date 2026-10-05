"""About — how it works, limitations, and dataset provenance."""
from __future__ import annotations

import streamlit as st

import ui


def render() -> None:
    bundle = ui.get_bundle()
    meta = bundle.meta
    b = meta.get("backends", {})

    ui.section_title("About", "How ScholarGrid works",
                     "An exploration tool, not a research-gap generator.")

    st.markdown("### The pipeline")
    st.markdown(f"""
1. **Collect** recent arXiv CS papers and clean them.
2. **Embed** title + abstract (`{b.get('embedding')}`) — the same method used for your searches.
3. **Reduce & cluster** (`{b.get('reduce')}` → `{b.get('cluster')}`) into research areas.
4. **Label** each area with distinctive keywords and representative papers.
5. **Project** everything to the 2D map (`{b.get('landscape')}`).
6. **Measure growth** vs. the whole corpus and **find sparse leads**.
7. **Validate** search, clusters, growth and robustness.
""")

    st.markdown("### What it is not")
    st.markdown("""
- The 2D map is a **picture, not proof** — distances can mislead.
- **Sparse ≠ novel.** A gap may just mean few papers were collected.
- **More papers ≠ better research.** Growth is activity, not quality.
- **Close in embedding space ≠ scientifically compatible.**

ScholarGrid helps you *find and inspect* areas worth a deeper look. It does
not decide what is valuable, novel, or publishable.
""")

    snap = meta.get("snapshot", {})
    st.markdown("### Dataset")
    ui.stat_tiles([
        (f"{meta['counts']['papers']:,}", "Papers"),
        (meta["counts"]["clusters"], "Areas"),
        (str(snap.get("earliest_included_date") or "—"), "From"),
        (str(snap.get("latest_included_date") or "—"), "To"),
    ])
    with st.expander("Full run metadata"):
        st.json(meta, expanded=False)
