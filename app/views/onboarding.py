"""First-run onboarding, shown when no artifact bundle exists yet."""
from __future__ import annotations

import streamlit as st

import ui
from scholargrid.config import load_config


def render() -> None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sg-eyebrow'>Welcome</div>", unsafe_allow_html=True)
        st.markdown("# 🔭 ScholarGrid")
        st.markdown("<p class='sg-lead'>There's no research landscape yet. "
                    "Build one to get started.</p>", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sg-card'><div class='t'>Try it now</div>"
                    "<div class='s'>Build a small offline demo landscape in a "
                    "few seconds — no internet needed.</div></div>",
                    unsafe_allow_html=True)
        if st.button("Load demo data", type="primary", use_container_width=True):
            _build_demo()

        st.markdown("<div class='sg-card'><div class='t'>Use real arXiv data</div>"
                    "<div class='s'>Harvest recent arXiv CS papers, then reload "
                    "this page.</div></div>", unsafe_allow_html=True)
        st.code("python pipeline/run_pipeline.py", language="bash")
        st.caption("Configure the window and sources in configs/config.yaml.")


def _build_demo() -> None:
    from scholargrid.runner import run

    demo = load_config()
    demo.raw["data"]["source"] = "synthetic"
    demo.raw["data"]["max_papers"] = 800
    bar = st.progress(0.0, "Starting…")
    run(demo, progress=lambda f, m: bar.progress(min(f, 1.0), m))
    st.cache_resource.clear()
    st.success("Demo landscape ready.")
    st.rerun()
