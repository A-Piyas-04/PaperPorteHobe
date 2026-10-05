"""Trends — publication growth across research areas."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui


def render() -> None:
    bundle = ui.get_bundle()
    clusters, growth = bundle.clusters_meta, bundle.growth
    dw = str(growth["default_window"])

    ui.section_title("Trends", "Growth across areas",
                     f"Relative growth over the last {dw} months vs. the whole "
                     "CS corpus. Above 1 means faster than average.")

    rows = []
    for cid, g in growth["clusters"].items():
        cid = int(cid)
        info = clusters.get(cid) or clusters.get(str(cid)) or {}
        w = g["windows"][dw]
        rows.append({
            "Area": info.get("label", f"cluster {cid}"),
            "Relative growth": round(w["relative_growth"], 2),
            "Recent papers": w["recent_count"],
            "Stable": "yes" if g["stability"]["stable"] else "varies",
        })
    table = pd.DataFrame(rows).sort_values("Relative growth", ascending=False)

    top = table.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["Relative growth"], y=top["Area"], orientation="h",
        marker=dict(color=ui.ACCENT), hovertemplate="%{y}<br>%{x}x<extra></extra>"))
    fig.update_layout(
        height=460, margin=dict(l=0, r=10, t=10, b=0),
        plot_bgcolor="#fbfcfd", paper_bgcolor="#fff",
        xaxis=dict(title="Relative growth", gridcolor="#eef0f3"),
        yaxis=dict(automargin=True))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### All areas")
    st.dataframe(table, use_container_width=True, hide_index=True)
