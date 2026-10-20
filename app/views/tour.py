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
# The walkthrough follows the research journey (Start → Topic → Read first →
# Open problems → Plan → My project), then the optional depth tools.
# ---------------------------------------------------------------------------
STEPS: list[dict] = [
    {
        "page": "start",
        "chapter": "Find your topic",
        "title": "Start with a topic",
        "body": "Describe what you want to research — a topic, a problem, or a method. ScholarGrid "
                "maps it using real papers; it does not write an answer for you.",
        "action": "Type a topic, or pick one of the examples, then choose “Explore this topic”.",
    },
    {
        "page": "topic",
        "chapter": "Find your topic",
        "title": "See its sub-topics",
        "body": "The papers on your topic are grouped into sub-topics with plain-language summaries, "
                "key concepts and their most common methods and datasets. Activity is based on the "
                "papers retrieved, not a growth estimate.",
        "action": "Open a sub-topic that looks interesting.",
    },
    {
        "page": "reading",
        "chapter": "Learn the area",
        "title": "Read in order",
        "body": "Start with an overview, then the foundations many papers here cite (often older than "
                "the collection), then key recent work and the frontier. Each paper says why it is there.",
        "action": "Save two or three papers, or export the path.",
    },
    {
        "page": "gaps",
        "chapter": "Find a question",
        "title": "Open problems, with evidence",
        "body": "Each card is built from what authors wrote: quotes you can check, a novelty check "
                "against recent papers, and “why this might not be a gap”. Treat every card as a lead.",
        "action": "Read a card's evidence, then save one you like.",
    },
    {
        "page": "plan",
        "chapter": "Find a question",
        "title": "Make a starter plan",
        "body": "Turn one open problem into a realistic first project for your time, goal and compute, "
                "grounded in the papers behind it.",
        "action": "Fill in “About you” and create a plan.",
    },
    {
        "page": "project",
        "chapter": "Keep it",
        "title": "Your project",
        "body": "Saved papers, open problems, plans and notes stay here after you close the tab. "
                "Bookmark the link, or export everything as JSON.",
        "action": "Change a paper's status or add a note.",
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
        "page": "trends",
        "chapter": "Go deeper when you need to",
        "title": "Where to look",
        "body": "Areas placed by how crowded they are and whether they are reliably growing, sized by "
                "their open problems, with plain advice for each kind of area.",
        "action": "Hover an area, or open the problems of one from the table.",
    },
]

TOTAL = len(STEPS)

_FALLBACK_TITLES = {
    "start": "Start", "topic": "Topic", "reading": "Read first", "gaps": "Open problems", "plan": "Plan",
    "project": "My project", "home": "Search papers", "explore": "Map", "areas": "All areas", "trends": "Trends",
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
            + "<p class='sg-tour-body'>Pick a topic, read in order, choose an open problem backed by "
              "evidence, and leave with a starter plan saved in your project. You can reopen this "
              "tour at any time from <b>Guide</b> in the header.</p>",
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

    with st.popover("Guide", width="content"):
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
            + f"<p class='sg-tour-body'>A {TOTAL}-step walkthrough of the whole journey, from a "
              "topic to a starter research plan. You can skip it at any point, and "
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
