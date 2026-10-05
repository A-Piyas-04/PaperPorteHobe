"""Investigation leads — sparse neighbourhoods near active areas."""
from __future__ import annotations

import streamlit as st

import ui


def render() -> None:
    bundle = ui.get_bundle()
    leads = bundle.sparse_leads

    ui.section_title("Leads", "Investigation leads",
                     "Low-density regions next to active areas, verified in high "
                     "dimensions and across projections. Leads to look into — not "
                     "proven gaps.")

    if not leads:
        st.markdown("<div class='sg-card'><div class='t'>No robust leads</div>"
                    "<div class='s'>The current landscape produced no sparse "
                    "neighbourhoods that survived the robustness checks.</div>"
                    "</div>", unsafe_allow_html=True)
        return

    cols = st.columns(2, gap="medium")
    for i, ld in enumerate(leads, 1):
        with cols[(i - 1) % 2]:
            _lead_card(i, ld)


def _lead_card(i: int, ld: dict) -> None:
    robustness = float(ld.get("robustness", 0))
    pct = int(round(robustness * 100))
    st.markdown(
        f"<div class='sg-card'>"
        f"<div class='t'>Lead {i} · near {ld.get('nearest_cluster_label') or 'unlabeled'}</div>"
        f"<div class='m'>Robustness {pct}%</div>"
        f"<div class='sg-meter'><span style='width:{pct}%'></span></div>"
        f"</div>", unsafe_allow_html=True)
    for e in ld.get("evidence_papers", [])[:3]:
        ui.paper_card(e["title"], e["arxiv_id"], f"{e['primary_category']} · {e['date']}")
