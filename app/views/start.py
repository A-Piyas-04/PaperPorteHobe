"""Start — the research journey's entry point: describe a topic, then map it."""
from __future__ import annotations

import streamlit as st

import ui

_STEPS = [
    ("1 · Topic", "See the sub-topics of your topic, in plain language."),
    ("2 · Read first", "An ordered reading path: overviews, foundations, recent work, the frontier."),
    ("3 · Open problems", "Problems authors say are still open, each backed by a quote."),
    ("4 · Plan", "Turn one problem into a starter research plan."),
    ("5 · Project", "Keep your topic, notes and plan between sessions."),
]


def _choose(text: str) -> None:
    text = " ".join(str(text).split())[: ui.max_query_chars()]
    if not text:
        return
    st.session_state.topic_query = text
    hist = [q for q in st.session_state.get("topic_history", []) if q != text]
    st.session_state.topic_history = [text, *hist][:5]
    st.session_state.pop("ws_selected", None)
    from telemetry import query_props, track

    track("topic_started", **query_props(ui.get_config(), text))


def render() -> None:
    bundle = ui.get_bundle()
    with ui.panel("hero"):
        st.markdown(
            "<div class='sg-eyebrow'>Start your research</div>"
            "<h1 class='sg-title'>What do you want to research?</h1>"
            "<p class='sg-lead'>Describe a topic, a problem, or a method. We'll map its sub-topics, tell you "
            "what to read first, and surface open problems backed by papers.</p>",
            unsafe_allow_html=True)
        with st.form("topic-form"):
            text = st.text_input("Your topic", key="topic_input", max_chars=ui.max_query_chars(),
                                 placeholder="e.g. hallucination in retrieval-augmented generation")
            submitted = st.form_submit_button("Explore this topic", type="primary", use_container_width=True)
        if submitted and text.strip():
            _choose(text)
            ui.goto("topic")
            st.rerun()
        st.caption("Or start with an example")
        cols = st.columns(3)
        for i, ex in enumerate(ui.topic_examples()[:3]):
            if cols[i].button(ex, key=f"topic_ex_{i}", use_container_width=True):
                _choose(ex)
                ui.goto("topic")
                st.rerun()

    st.markdown("<div class='sg-journey five'>" + "".join(
        f"<div><b>{t}</b><p>{d}</p></div>" for t, d in _STEPS) + "</div>", unsafe_allow_html=True)

    history = st.session_state.get("topic_history", [])
    if history:
        with ui.panel("recent-topics"):
            ui.panel_header("Recent topics", "this session")
            for i, q in enumerate(history[:5]):
                if st.button(q, key=f"topic_hist_{i}", use_container_width=True):
                    _choose(q)
                    ui.goto("topic")
                    st.rerun()

    cfg = ui.get_config()
    scope = cfg["scope"]["name"]
    live = cfg["search"]["live"] and cfg["workspace"]["live"]
    st.caption(f"Covers {len(bundle.df):,} analysed papers"
               + (f" ({scope})" if scope and scope != "all" else "")
               + (" plus live arXiv/OpenAlex/Semantic Scholar results" if live else "")
               + ". Open problems come from what authors wrote in their papers.")
