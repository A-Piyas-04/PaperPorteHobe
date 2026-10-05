"""First-run onboarding, shown when no artifact bundle exists yet."""
from __future__ import annotations

import streamlit as st

import ui
from scholargrid.config import load_config


def render() -> None:
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
        ui.page_header("Welcome", "🔭 ScholarGrid",
                       "There's no research landscape yet. Build one to get started.")

        with ui.panel("demo"):
            ui.panel_header("Try it now", "a few seconds")
            st.markdown("Build a small demo landscape offline. No internet needed.")
            if st.button("Load demo data", type="primary", use_container_width=True):
                _build_demo()

        with ui.panel("real"):
            ui.panel_header("Use real arXiv data")
            st.markdown("Harvest recent arXiv papers, then reload this page.")
            st.code("python pipeline/run_pipeline.py", language="bash")
            st.caption("Settings live in configs/config.yaml.")


def _build_demo() -> None:
    from scholargrid.runner import run

    demo = load_config()
    demo.raw["data"]["source"] = "synthetic"
    demo.raw["data"]["max_papers"] = 800
    bar = st.progress(0.0, "Starting…")
    run(demo, progress=lambda f, m: bar.progress(min(f, 1.0), m))
    st.cache_resource.clear()
    st.rerun()
