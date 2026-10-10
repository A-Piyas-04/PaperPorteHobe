"""A skippable, repeatable introduction to the research workflow."""
import streamlit as st

import ui

_STEPS = [
    ("Start with a question", "Describe what you want to understand in the Find papers search box. "
     "Try a topic, a method, or a problem such as efficient transformer inference. "
     "You will get a ranked reading list, not a generated answer."),
    ("See the landscape on the map", "Open the Workspace to explore the connected map. "
     "Start at the overview of research areas, click an area to reveal its papers and links, "
     "then click a paper to focus its connections. Use Overview to zoom back out. "
     "Prefer a plain list? Switch to the List view at any time."),
    ("Read the evidence", "Click a paper on the map or in the list to read its abstract and follow related papers. "
     "Hovering only previews a title; your selection stays put until you click another. Open arXiv for the original paper."),
    ("Build your next reading list", "Save useful papers from search results or the workspace to your Reading list and export them. "
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
