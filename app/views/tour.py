"""A skippable, repeatable introduction to the research workflow."""
import streamlit as st

import ui

_STEPS = [
    ("Start with a question", "Describe what you want to understand in the Find papers search box. "
     "Try a topic, a method, or a problem such as efficient transformer inference. "
     "You will get a ranked reading list, not a generated answer."),
    ("Put the papers in context", "Open Explore results to see your matches in the workspace. "
     "Filter by research area, or browse area cards if you are still choosing a topic."),
    ("Read the evidence", "Choose a paper to read its abstract and follow related papers. "
     "Your selection stays put until you click another title. Open arXiv for the original paper."),
    ("Build your next reading list", "Save useful papers from search results to your Reading list and export them. "
     "Use Trends and Investigation leads when you want more context. Signals depend on the dataset; "
     "a quiet area is not proof of a research gap."),
]


def render() -> None:
    ui.ss("tour_step", 0)
    ui.ss("tour_done", False)
    if st.session_state.tour_done:
        if st.button("Take the quick tour", key="restart_tour", type="tertiary"):
            st.session_state.tour_done = False
            st.session_state.tour_step = 0
            st.rerun()
        return
    step = st.session_state.tour_step
    with ui.panel("welcome-tour"):
        st.caption(f"QUICK TOUR · {step + 1} OF {len(_STEPS)}")
        st.subheader(_STEPS[step][0])
        st.write(_STEPS[step][1])
        st.progress((step + 1) / len(_STEPS))
        back, nxt, skip = st.columns([1, 1, 1])
        if back.button("Back", disabled=step == 0, use_container_width=True):
            st.session_state.tour_step -= 1
            st.rerun()
        if nxt.button("Start exploring" if step == len(_STEPS) - 1 else "Next", type="primary", use_container_width=True):
            if step == len(_STEPS) - 1:
                st.session_state.tour_done = True
            else:
                st.session_state.tour_step += 1
            st.rerun()
        if skip.button("Skip tour", use_container_width=True):
            st.session_state.tour_done = True
            st.rerun()
