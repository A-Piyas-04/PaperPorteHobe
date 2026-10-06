"""Trends — publication growth across research areas.

Growth numbers are shown only for areas that pass every gate (enough history,
enough papers in both periods, a confidence interval, stable across reference
dates). Every number carries its window, absolute counts and interval. When
the corpus is too short, the page says so and shows activity by paper count.
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


def _fmt_ci(ci) -> str:
    return f"{ci[0]:.2f}–{ci[1]:.2f}" if ci else "—"


def _growth(bundle) -> None:
    clusters, growth = bundle.clusters_meta, bundle.growth
    dw = str(growth["default_window"])
    level = int(round(float((growth.get("gate") or {}).get("ci_level", 0.9)) * 100))
    reliable, other = [], []
    for cid, g in growth["clusters"].items():
        w = g["windows"][dw]
        row = {
            "Area": ui.area_name(clusters, int(cid)),
            "Recent papers": w["recent_count"],
            "Previous papers": w["previous_count"],
        }
        if ui.growth_reliable(growth, cid) and w.get("relative_growth") is not None:
            row.update({"Growth": w["relative_growth"], f"{level}% CI": _fmt_ci(w.get("ci")),
                        "Signal": w.get("direction") or "—", "ci": w.get("ci")})
            reliable.append(row)
        else:
            reason = ("too few papers" if not w.get("passes_gate")
                      else "unstable across dates" if not g.get("stability", {}).get("stable")
                      else "not enough evidence")
            row.update({"Signal": f"trend not reliable ({reason})"})
            other.append(row)
    reliable.sort(key=lambda r: -r["Growth"])

    ui.note(f"<b>How to read this.</b> Growth compares the last {dw} months with the {dw} months "
            f"before, relative to all of CS (1.0× = same pace). An area is called <i>growing</i> "
            f"only when the whole {level}% confidence interval is above 1.0×. "
            f"{len(reliable)} of {len(reliable) + len(other)} areas have a reliable signal.")

    if reliable:
        with ui.panel("growth"):
            ui.panel_header("Reliable growth signals", f"last {dw} months vs. all of CS")
            top = reliable[:15][::-1]
            fig = go.Figure(go.Bar(
                x=[r["Growth"] for r in top], y=[r["Area"] for r in top], orientation="h",
                error_x=dict(type="data", symmetric=False,
                             array=[(r["ci"][1] - r["Growth"]) if r["ci"] else 0 for r in top],
                             arrayminus=[(r["Growth"] - r["ci"][0]) if r["ci"] else 0 for r in top],
                             color="#8a93a3"),
                marker=dict(color=[ui.ACCENT if r["Signal"] == "growing" else "#a9b4c6" for r in top]),
                customdata=[[r[f"{level}% CI"], r["Recent papers"], r["Previous papers"]] for r in top],
                hovertemplate="%{y}<br>%{x:.2f}× the CS pace (CI %{customdata[0]})"
                              "<br>%{customdata[1]} recent vs %{customdata[2]} previous papers<extra></extra>"))
            ui.apply_chart_style(fig, height=max(320, 34 * len(top) + 80))
            fig.add_vline(x=1, line_dash="dot", line_color="#8a93a3")
            fig.update_layout(xaxis=dict(title="Growth vs. CS average", gridcolor="#eef0f3"),
                              yaxis=dict(automargin=True, tickfont=dict(size=15)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with ui.panel("growth-table"):
            ui.panel_header("Areas with a reliable signal")
            for r in reliable:
                r["Growth"] = f"{r['Growth']:.2f}×"
            ui.html_table(reliable, ["Area", "Growth", f"{level}% CI", "Recent papers",
                                     "Previous papers", "Signal"],
                          numeric=("Growth", "Recent papers", "Previous papers"), bar_col="Recent papers")

    if other:
        with ui.panel("growth-counts"):
            ui.panel_header("Counts only", f"{len(other)} areas")
            other.sort(key=lambda r: -r["Recent papers"])
            ui.html_table(other, ["Area", "Recent papers", "Previous papers", "Signal"],
                          numeric=("Recent papers", "Previous papers"), bar_col="Recent papers")
