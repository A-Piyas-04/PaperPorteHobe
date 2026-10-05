"""Trends — publication growth across research areas.

When the corpus is too short for window-based growth, the page says so and
shows the most active areas by paper count instead of misleading ratios.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import ui


def render() -> None:
    bundle = ui.get_bundle()
    ready = ui.growth_ready(bundle)

    ui.page_header("Trends", "Where research is moving",
                   "Which areas are publishing the most, and which are speeding up.")

    if ready:
        _growth(bundle)
    else:
        _activity(bundle)


def _activity(bundle) -> None:
    clusters, colors = bundle.clusters_meta, ui.area_colors(bundle.clusters_meta)
    days = ui.history_days(bundle)
    window = int(bundle.growth["default_window"])
    ui.note(f"<b>Not enough history for growth yet.</b> The data covers {days} days; "
            f"comparing {window}-month periods needs at least {2 * window} months. "
            "Showing the most active areas instead.")

    items = sorted(clusters.values(), key=lambda c: -c.get("size", 0))
    with ui.panel("activity"):
        ui.panel_header("Most active areas", f"{len(items)} areas")
        top = items[:12][::-1]
        fig = go.Figure(go.Bar(
            x=[c["size"] for c in top],
            y=[ui.area_name(clusters, c["cluster_id"]) for c in top],
            orientation="h",
            marker=dict(color=[colors[int(c["cluster_id"])] for c in top]),
            hovertemplate="%{y}<br>%{x} papers<extra></extra>"))
        ui.apply_chart_style(fig, height=520)
        fig.update_layout(xaxis=dict(title="Papers", gridcolor="#eef0f3"),
                          yaxis=dict(automargin=True, tickfont=dict(size=15)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with ui.panel("activity-table"):
        ui.panel_header("All areas")
        total = sum(c.get("size", 0) for c in items) or 1
        rows = [{"Area": ui.area_name(clusters, c["cluster_id"]),
                 "Papers": c.get("size", 0),
                 "Share": f"{c.get('size', 0) / total:.0%}"} for c in items]
        ui.html_table(rows, ["Area", "Papers", "Share"], numeric=("Papers", "Share"),
                      bar_col="Papers")


def _growth(bundle) -> None:
    clusters, growth = bundle.clusters_meta, bundle.growth
    dw = str(growth["default_window"])
    rows = []
    for cid, g in growth["clusters"].items():
        w = g["windows"][dw]
        rows.append({
            "Area": ui.area_name(clusters, int(cid)),
            "Growth": round(w["relative_growth"], 2),
            "Recent papers": w["recent_count"],
            "Signal": "steady" if g["stability"]["stable"] else "varies",
        })
    rows.sort(key=lambda r: -r["Growth"])

    with ui.panel("growth"):
        ui.panel_header("Fastest growing", f"last {dw} months vs. all of CS")
        top = rows[:12][::-1]
        fig = go.Figure(go.Bar(
            x=[r["Growth"] for r in top], y=[r["Area"] for r in top], orientation="h",
            marker=dict(color=[ui.ACCENT if r["Growth"] >= 1 else "#a9b4c6" for r in top]),
            hovertemplate="%{y}<br>%{x}× the CS average<extra></extra>"))
        ui.apply_chart_style(fig, height=520)
        fig.add_vline(x=1, line_dash="dot", line_color="#8a93a3")
        fig.update_layout(xaxis=dict(title="Growth vs. CS average", gridcolor="#eef0f3"),
                          yaxis=dict(automargin=True, tickfont=dict(size=15)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with ui.panel("growth-table"):
        ui.panel_header("All areas")
        for r in rows:
            r["Growth"] = f"{r['Growth']:.2f}×"
        ui.html_table(rows, ["Area", "Growth", "Recent papers", "Signal"],
                      numeric=("Growth", "Recent papers"), bar_col="Recent papers")
