"""Guided tour — a page-aware, skippable walkthrough of the research workflow.

The tour is an ordered list of steps, each bound to a page. ``render_dock``
draws the current step above the page content on every page; ``header_control``
puts start / resume / restart / end into the top navigation, so the tour is
reachable from anywhere. Pressing Next advances and, when the next step lives on
another page, navigates there — so the tour reads as "do this, then this".

Nothing here blocks the app: every state has a Skip, and the dock disappears the
moment the tour is inactive.
"""
from __future__ import annotations

import html

import streamlit as st

import ui

# ---------------------------------------------------------------------------
# The walkthrough. Four chapters following the app's own 01-02-03 journey, with
# the optional depth tools sitting between "understand" and "leave with it".
# ---------------------------------------------------------------------------
STEPS: list[dict] = [
    {
        "page": "home",
        "chapter": "Find your starting point",
        "title": "Start with a question",
        "body": "Describe what you want to understand — a topic, a method, or a problem. "
                "ScholarGrid ranks real papers against your words. It does not write an "
                "answer for you, so what you get back is a reading list you can check.",
        "action": "Type a question into the search box, or pick one of the examples.",
    },
    {
        "page": "home",
        "chapter": "Find your starting point",
        "title": "Read the results",
        "body": "Exact matches contain your words. Related papers are close in meaning but "
                "may use different terms. Each card shows authors, research area, date and "
                "citations, so you can judge a paper before you open it.",
        "action": "Open “Filters and sorting” to narrow by date, category or citations.",
    },
    {
        "page": "home",
        "chapter": "Find your starting point",
        "title": "Save as you go",
        "body": "Keep anything worth reading. “Save to reading list” stores a paper for this "
                "session, and “More like this” searches for work similar to it — useful when "
                "one good paper is a better query than your original words.",
        "action": "Save one paper, then carry on reading.",
    },
    {
        "page": "explore",
        "chapter": "Understand the context",
        "title": "See the landscape",
        "body": "The Workspace map shows how research areas sit next to each other. Each bubble "
                "is an area, bigger means more papers, and a link means two areas are textually "
                "similar. Position comes from an embedding projection, so distance is a rough "
                "guide to similarity — not a measure of research value.",
        "action": "Click a bubble to open that area's papers.",
    },
    {
        "page": "explore",
        "chapter": "Understand the context",
        "title": "Follow the connections",
        "body": "Inside an area every node is a paper. Click one to read its abstract and related "
                "work in the side panel; your choice stays selected until you pick another. "
                "Hovering only previews a title. Overview and Fit take you back out.",
        "action": "Click a paper, then try “Focus connections”. Prefer plain text? Switch to List.",
    },
    {
        "page": "areas",
        "chapter": "Go deeper when you need to",
        "title": "Browse research areas",
        "body": "Areas are groups of papers with similar titles and abstracts, named after their "
                "keywords. Open one to see its representative papers and how active it has been "
                "month by month.",
        "action": "Filter by a keyword, then open an area that looks relevant.",
    },
    {
        "page": "trends",
        "chapter": "Go deeper when you need to",
        "title": "See where research is moving",
        "body": "Growth compares recent publishing against an earlier period. A number only "
                "appears when the data supports it — otherwise the page says so and shows plain "
                "activity instead. Every figure carries its window and interval.",
        "action": "Compare a few areas, and read the window and interval under each number.",
    },
    {
        "page": "leads",
        "chapter": "Go deeper when you need to",
        "title": "Treat leads as questions",
        "body": "Leads are quiet spots sitting next to active areas. They can hint at an "
                "under-explored direction, but just as often they reflect what was collected or "
                "how the map was drawn. A lead is a question to check, never proof of a gap.",
        "action": "Open a lead and read the papers around it before trusting it.",
    },
    {
        "page": "library",
        "chapter": "Leave with something",
        "title": "Export your reading list",
        "body": "Everything you saved lives here. Download it as BibTeX for a reference manager, "
                "or CSV for a spreadsheet. The list is kept for this session only, so export it "
                "before you close the tab.",
        "action": "Download your list — or go back and save a few more papers first.",
    },
]

TOTAL = len(STEPS)

_FALLBACK_TITLES = {
    "home": "Search papers", "explore": "Map", "areas": "All areas",
    "trends": "Trends", "leads": "Leads", "library": "Reading list",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def _init() -> None:
    ui.ss("tour_step", 0)
    ui.ss("tour_active", False)
    ui.ss("tour_done", False)


def _page_title(key: str) -> str:
    page = (st.session_state.get("_pages") or {}).get(key)
    return getattr(page, "title", None) or _FALLBACK_TITLES.get(key, key)


def step_for_page(page_key: str | None) -> int | None:
    """Index of the first step belonging to ``page_key``, if any."""
    if not page_key:
        return None
    for i, s in enumerate(STEPS):
        if s["page"] == page_key:
            return i
    return None


def start(index: int = 0) -> None:
    st.session_state.tour_step = max(0, min(index, TOTAL - 1))
    st.session_state.tour_active = True


def pause() -> None:
    """Leave the tour but remember where we were, so Resume can continue."""
    st.session_state.tour_active = False
    st.session_state.tour_done = True


def finish() -> None:
    st.session_state.tour_active = False
    st.session_state.tour_done = True
    st.session_state.tour_step = 0


def _go(index: int) -> None:
    """Move to a step, following it to its page when that differs from here."""
    if index >= TOTAL:
        st.session_state.tour_step = TOTAL        # completion card
        st.rerun()
    index = max(0, index)
    st.session_state.tour_step = index
    target = STEPS[index]["page"]
    if target != st.session_state.get("_page_key"):
        ui.goto(target)                           # switches page (and reruns)
    st.rerun()


# ---------------------------------------------------------------------------
# The dock (rendered above page content on every page)
# ---------------------------------------------------------------------------
def render_dock() -> None:
    _init()
    if not st.session_state.tour_active:
        return
    idx = int(st.session_state.tour_step)
    if idx >= TOTAL:
        _complete()
        return
    step = STEPS[idx]
    current = st.session_state.get("_page_key")
    with st.container(key="tour-dock"):
        if current and current != step["page"]:
            _offpage(idx, step, current)
        else:
            _step(idx, step)


def _dots(idx: int) -> str:
    """Segmented step rail — the tour's own progress indicator, not Streamlit's."""
    cells = "".join(
        f"<i class='{'now' if i == idx else 'done' if i < idx else ''}'></i>"
        for i in range(TOTAL))
    return f"<div class='sg-tour-dots'>{cells}</div>"


def _head(count: str) -> str:
    return (f"<div class='sg-tour-head'><span class='sg-tour-tag'>Guided tour</span>"
            f"<span class='sg-tour-count'>{html.escape(count)}</span></div>")


def _step(idx: int, step: dict) -> None:
    st.markdown(
        _head(f"Step {idx + 1} of {TOTAL} · {step['chapter']}")
        + f"<div class='sg-tour-title'>{html.escape(step['title'])}</div>"
        + f"<p class='sg-tour-body'>{html.escape(step['body'])}</p>"
        + f"<div class='sg-tour-do'>▸ {html.escape(step['action'])}</div>"
        + _dots(idx),
        unsafe_allow_html=True)
    c1, c2, c3, _ = st.columns([1, 1.5, 1, 2.5])
    if c1.button("Back", key="tour_back", disabled=idx == 0, use_container_width=True):
        _go(idx - 1)
    if c2.button("Finish" if idx == TOTAL - 1 else "Next",
                 key="tour_next", type="primary", use_container_width=True):
        _go(idx + 1)
    if c3.button("Skip tour", key="tour_skip", use_container_width=True):
        pause()
        st.rerun()


def _offpage(idx: int, step: dict, current: str) -> None:
    """The user navigated somewhere else while the tour was running."""
    target = _page_title(step["page"])
    here = step_for_page(current)
    st.markdown(
        _head(f"Step {idx + 1} of {TOTAL} · paused")
        + f"<div class='sg-tour-title'>Next up: {html.escape(step['title'])}</div>"
        + f"<p class='sg-tour-body'>That step is on <b>{html.escape(target)}</b>. Carry on there, "
          f"or pick the tour up where you are now.</p>",
        unsafe_allow_html=True)
    c1, c2, c3, _ = st.columns([1.5, 1.8, 1, 2])
    if c1.button(f"Go to {target}", key="tour_resume_there", type="primary",
                 use_container_width=True):
        _go(idx)
    if here is not None and c2.button("Continue from this page", key="tour_resume_here",
                                      use_container_width=True):
        _go(here)
    if c3.button("Skip tour", key="tour_skip_off", use_container_width=True):
        pause()
        st.rerun()


def _complete() -> None:
    with st.container(key="tour-dock"):
        st.markdown(
            _head("Complete")
            + "<div class='sg-tour-title'>That’s the whole workflow</div>"
            + "<p class='sg-tour-body'>Search for an interest, read the papers around it on the "
              "map, and leave with an exported reading list. You can reopen this tour at any "
              "time from <b>Guide</b> in the header.</p>",
            unsafe_allow_html=True)
        c1, c2, _ = st.columns([1.4, 1.3, 3])
        if c1.button("Start exploring", key="tour_finish", type="primary",
                     use_container_width=True):
            finish()
            st.rerun()
        if c2.button("Restart tour", key="tour_restart_done", use_container_width=True):
            start(0)
            _go(0)


# ---------------------------------------------------------------------------
# Header control (top navigation, every page)
# ---------------------------------------------------------------------------
def header_control() -> None:
    _init()
    active = bool(st.session_state.tour_active)
    done = bool(st.session_state.tour_done)
    idx = int(st.session_state.tour_step)
    here = step_for_page(st.session_state.get("_page_key"))

    with st.popover("Guide", use_container_width=True):
        if active:
            st.caption("Guided tour · finished" if idx >= TOTAL
                       else f"Guided tour · step {idx + 1} of {TOTAL}")
        elif done and idx > 0:
            st.caption(f"Guided tour · paused at step {min(idx + 1, TOTAL)} of {TOTAL}")
        else:
            st.caption(f"A {TOTAL}-step walkthrough of how to use ScholarGrid.")

        if not active:
            resuming = done and idx > 0
            if st.button("Resume tour" if resuming else "Take the guided tour",
                         key="tour_hdr_start", type="primary", use_container_width=True):
                start(idx if resuming else 0)
                _go(st.session_state.tour_step)
        if here is not None and st.button("Start from this page", key="tour_hdr_here",
                                          use_container_width=True):
            start(here)
            st.rerun()
        if st.button("Restart from the beginning", key="tour_hdr_restart",
                     use_container_width=True):
            start(0)
            _go(0)
        if active and st.button("End tour", key="tour_hdr_end", use_container_width=True):
            pause()
            st.rerun()
        st.caption("You can skip or restart at any point.")


# ---------------------------------------------------------------------------
# Landing-page invitation (first visit only)
# ---------------------------------------------------------------------------
def render_invite() -> None:
    _init()
    if st.session_state.tour_active or st.session_state.tour_done:
        return
    with st.container(key="tour-invite"):
        st.markdown(
            _head("First time here?")
            + "<div class='sg-tour-title'>Take the guided tour</div>"
            + f"<p class='sg-tour-body'>A {TOTAL}-step walkthrough of the whole workflow, from a "
              "first search to an exported reading list. You can skip it at any point, and "
              "reopen it later from <b>Guide</b> in the header.</p>",
            unsafe_allow_html=True)
        c1, c2, _ = st.columns([1.4, 1.1, 2.5])
        if c1.button("Take the guided tour", key="tour_invite_start", type="primary",
                     use_container_width=True):
            start(0)
            _go(0)
        if c2.button("No thanks", key="tour_invite_skip", use_container_width=True):
            st.session_state.tour_done = True
            st.rerun()
