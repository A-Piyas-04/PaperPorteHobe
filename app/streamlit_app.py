"""ScholarGrid — Research Landscape Explorer (Streamlit web app).

A lightweight, multi-page front-end over precomputed artifacts (NFR-01): only
live query embedding + nearest-neighbour retrieval happen here. Heavy analysis
is done offline by the pipeline.

This module is a thin shell: it sets up the page, injects the global
stylesheet, loads the artifact bundle once, and wires the pages together with a
custom top navigation bar. Each page lives in ``app/views/``.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ui  # noqa: E402  (app/ui.py)
from scholargrid.artifacts import bundle_exists  # noqa: E402
from scholargrid.config import load_config  # noqa: E402

st.set_page_config(page_title="ScholarGrid", page_icon="🔭", layout="wide")


def _build_pages() -> dict:
    """Create the st.Page objects, keyed by a short name for programmatic nav."""
    from views import about, areas, explore, home, leads, trends

    return {
        "home": st.Page(home.render, title="Search", icon=":material/search:",
                        url_path="home", default=True),
        "explore": st.Page(explore.render, title="Explore", icon=":material/hub:",
                           url_path="explore"),
        "areas": st.Page(areas.render, title="Areas", icon=":material/category:",
                         url_path="areas"),
        "trends": st.Page(trends.render, title="Trends", icon=":material/trending_up:",
                          url_path="trends"),
        "leads": st.Page(leads.render, title="Leads", icon=":material/lightbulb:",
                         url_path="leads"),
        "about": st.Page(about.render, title="About", icon=":material/info:",
                         url_path="about"),
    }


def main() -> None:
    ui.inject_css()
    cfg = load_config()

    # First run: no artifacts yet -> standalone onboarding, no nav.
    if not bundle_exists(cfg):
        from views import onboarding
        onboarding.render()
        return

    ui.ensure_loaded()

    pages = _build_pages()
    st.session_state["_pages"] = pages

    nav = st.navigation(list(pages.values()), position="hidden")
    ui.render_top_nav(pages)
    nav.run()


if __name__ == "__main__":
    main()
