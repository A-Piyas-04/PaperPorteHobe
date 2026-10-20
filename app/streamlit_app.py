"""ScholarGrid — Research Landscape Explorer (Streamlit web app).

A lightweight, multi-page front-end over precomputed artifacts (NFR-01): only
live query embedding + nearest-neighbour retrieval happen here. Heavy analysis
is done offline by the pipeline.

This module is a thin shell: it sets up the page, injects the global
stylesheet, loads the artifact bundle once, and wires the pages together with a
custom top navigation bar. Each page lives in ``app/views/``. Errors inside a
page are reported (Sentry when configured) and shown as a friendly message,
never as a raw traceback.

Run:  streamlit run app/streamlit_app.py
Health check: Streamlit serves ``/_stcore/health``.
"""
from __future__ import annotations

import os
import sys

import streamlit as st

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_APP_DIR))
sys.path.insert(0, _APP_DIR)  # app.py (Spaces shim) runs this file with the repo root as cwd

import ui  # noqa: E402  (app/ui.py)
from scholargrid.artifacts import bundle_exists  # noqa: E402
from scholargrid.monitoring import capture_exception, init_error_tracking  # noqa: E402

st.set_page_config(page_title="ScholarGrid", page_icon="🔭", layout="wide")


def _build_pages() -> dict:
    """Create the st.Page objects, keyed by a short name for programmatic nav."""
    from views import (
        about,
        areas,
        explore,
        gaps,
        home,
        leads,
        library,
        plan,
        project,
        reading,
        start,
        topic,
        trends,
    )

    return {
        "start": st.Page(start.render, title="Start", icon=":material/flag:", url_path="start", default=True),
        "topic": st.Page(topic.render, title="Topic", icon=":material/account_tree:", url_path="topic"),
        "reading": st.Page(reading.render, title="Read first", icon=":material/menu_book:",
                           url_path="read-first"),
        "gaps": st.Page(gaps.render, title="Open problems", icon=":material/explore:", url_path="open-problems"),
        "plan": st.Page(plan.render, title="Plan", icon=":material/route:", url_path="plan"),
        "project": st.Page(project.render, title="My project", icon=":material/folder_open:", url_path="project"),
        "library": st.Page(library.render, title="Reading list", icon=":material/bookmarks:",
                           url_path="reading-list"),
        "home": st.Page(home.render, title="Search papers", icon=":material/search:", url_path="search"),
        "explore": st.Page(explore.render, title="Map", icon=":material/hub:", url_path="explore"),
        "areas": st.Page(areas.render, title="All areas", icon=":material/category:", url_path="areas"),
        "trends": st.Page(trends.render, title="Trends", icon=":material/trending_up:", url_path="trends"),
        "leads": st.Page(leads.render, title="Leads (legacy)", icon=":material/lightbulb:", url_path="leads"),
        "about": st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
    }


def _current_key(pages: dict, nav) -> str | None:
    """Map the running page back to its short key (``home``, ``explore``, …)."""
    path = getattr(nav, "url_path", None)
    for key, page in pages.items():
        if getattr(page, "url_path", None) == path:
            return key
    return None


def _friendly_error(exc: Exception) -> None:
    capture_exception(exc)
    ui.note("<b>Something went wrong on this page.</b> The error has been logged. "
            "Try again, or open another page.")


def main() -> None:
    init_error_tracking("app")
    ui.inject_css()
    cfg = ui.get_config()

    # First run: no artifacts yet -> standalone onboarding, no nav.
    if not bundle_exists(cfg):
        from views import onboarding
        onboarding.render()
        return

    try:
        ui.ensure_loaded()
    except Exception as exc:
        _friendly_error(exc)
        return

    pages = _build_pages()
    st.session_state["_pages"] = pages
    # Keep the last query when Streamlit cleans up widgets on other pages.
    for key in ("q", "topic_input"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]

    nav = st.navigation(list(pages.values()), position="hidden")
    # Which page is showing, as a short key — the guided tour binds steps to these.
    st.session_state["_page_key"] = _current_key(pages, nav)

    from views import tour

    ui.render_top_nav(pages)
    ui.global_notices(ui.get_bundle())
    if st.session_state.get("_page_key") != "project":
        ui.project_notices()
    tour.render_dock()
    try:
        nav.run()
    except Exception as exc:
        _friendly_error(exc)
    ui.attribution_footer()


if __name__ == "__main__":
    main()
