"""Investigation leads — sparse neighbourhoods near active areas."""
from __future__ import annotations

import html

import streamlit as st

import ui


def render() -> None:
    bundle = ui.get_bundle()
    leads, clusters = bundle.sparse_leads, bundle.clusters_meta

    ui.page_header("Leads", "Investigation leads",
                   "Quiet spots right next to busy areas. Worth a look, not proven gaps.")

    if not leads:
        with ui.panel("no-leads"):
            ui.panel_header("No robust leads")
            st.markdown("The current landscape has no sparse spots that passed the checks.")
        return

    cols = st.columns(2, gap="medium")
    for i, ld in enumerate(leads, 1):
        with cols[(i - 1) % 2], ui.panel(f"lead-{i}"):
            _lead(i, ld, clusters)


def _lead(i: int, ld: dict, clusters: dict) -> None:
    pct = int(round(float(ld.get("robustness", 0)) * 100))
    near = ld.get("nearest_cluster_id")
    near_name = (ui.area_name(clusters, near) if near is not None
                 else ld.get("nearest_cluster_label") or "unlabeled")
    ui.panel_header(f"Lead {i}", f"{pct}% robust")
    st.markdown(f"<div class='sg-area-m'>Next to <b>{html.escape(near_name)}</b></div>"
                f"<div class='sg-meter'><span style='width:{pct}%'></span></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    for n, e in enumerate(ld.get("evidence_papers", [])[:3]):
        ui.paper_card(e["title"], e["arxiv_id"], category=e["primary_category"],
                      date=e["date"], delay=n)
