"""Admin — product funnel, quality against the launch bar, 'not useful' reasons, free quota, data freshness."""
from __future__ import annotations

import time
from collections import Counter, defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

_FUNNEL = ["topic_started", "reading_viewed", "gaps_generated", "gap_saved", "plan_generated", "project_returned"]
_FUNNEL_LABELS = {"topic_started": "Started a topic", "reading_viewed": "Viewed Read first",
                  "gaps_generated": "Generated open problems", "gap_saved": "Saved an open problem",
                  "plan_generated": "Made a plan", "project_returned": "Came back to a project (7 days)"}


def _funnel(events: list[dict]) -> list[tuple[str, int]]:
    by_session: dict = defaultdict(dict)
    project_first: dict = {}
    for e in events:
        ts = pd.Timestamp(e["ts"]).timestamp() if e.get("ts") else 0.0
        by_session[e["session"]].setdefault(e["name"], ts)
        if e.get("project_id") and e["name"] in ("project_created", "gap_saved", "paper_saved"):
            project_first.setdefault(e["project_id"], ts)
    returned = {e["project_id"] for e in events if e["name"] == "project_returned" and e.get("project_id")
                and pd.Timestamp(e["ts"]).timestamp() - project_first.get(e["project_id"], 0) <= 7 * 86400}
    sessions_returning = {e["session"] for e in events if e["name"] == "project_returned"
                          and e.get("project_id") in returned}
    out, alive = [], set(by_session)
    for step in _FUNNEL:
        if step == "project_returned":
            alive = alive | sessions_returning
            n = len(sessions_returning)
        else:
            alive = {s for s in alive if step in by_session[s]}
            n = len(alive)
        out.append((_FUNNEL_LABELS[step], n))
    return out


def render() -> None:
    from scholargrid.evaluation.metrics import LAUNCH_BAR, latest_history

    if not ui.eval_allowed():
        ui.note("This page is only available in evaluation mode.")
        return
    cfg = ui.get_config()
    repo = ui.get_repo()
    bundle = ui.get_bundle()
    ui.page_header("Admin", "How ScholarGrid is doing", "Product funnel, measured quality and free-quota use.")

    with ui.panel("admin-funnel"):
        ui.panel_header("Funnel", "last 30 days")
        events = repo.events(30) if repo is not None else []
        if not events:
            st.info("No telemetry events yet.")
        else:
            steps = _funnel(events)
            fig = go.Figure(go.Funnel(y=[s for s, _ in steps], x=[n for _, n in steps]))
            ui.apply_chart_style(fig, height=320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            reasons = Counter(e["props"].get("reason", "Other") for e in events if e["name"] == "gap_not_useful")
            if reasons:
                st.markdown("<div class='sg-section'>“Not useful” reasons</div>", unsafe_allow_html=True)
                fig2 = go.Figure(go.Bar(x=list(reasons.values()), y=list(reasons.keys()), orientation="h"))
                ui.apply_chart_style(fig2, height=240)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                recent = [e for e in events if e["name"] == "gap_not_useful"][-10:]
                st.markdown("\n".join(f"- `{e['props'].get('gap_id')}` ({e['props'].get('type')}): "
                                      f"{e['props'].get('reason')}" for e in reversed(recent)))

    with ui.panel("admin-quality"):
        ui.panel_header("Quality", "latest rated rounds")
        rows = []
        for kind in ("gaps", "extraction", "plans", "search", "reading"):
            last = latest_history(cfg, kind)
            if not last:
                continue
            for metric, (op, thr) in LAUNCH_BAR.items():
                if metric in last and last[metric] is not None:
                    v = float(last[metric])
                    ok = v >= thr if op == ">=" else v <= thr
                    rows.append({"Round": last.get("batch"), "Metric": metric, "Value": round(v, 3),
                                 "Bar": f"{op} {thr}", "OK": "yes" if ok else "no"})
        if rows:
            ui.html_table(rows, ["Round", "Metric", "Value", "Bar", "OK"], numeric=("Value",))
        else:
            st.info("No rated rounds yet.")

    with ui.panel("admin-quota"):
        ui.panel_header("Free LLM quota")
        from scholargrid.llm.quota import QuotaManager
        from scholargrid.llm.usage import read_usage

        lc = cfg["llm"]
        q = QuotaManager(cfg.abspath(lc["usage_dir"]), lc["limits"], {n: p["quota_tz"] for n, p in lc["providers"].items()})
        routes = sorted({r for t in lc["tasks"].values() for r in t["route"]})
        ui.html_table([{"Route": r, "Used today": q.used_today(r), "Left today": q.remaining_today(r)
                        if q.remaining_today(r) is not None else "?",
                        "Exhausted until": time.strftime("%Y-%m-%d %H:%M", time.localtime(q.exhausted_until(r)))
                        if q.exhausted_until(r) else ""} for r in routes],
                      ["Route", "Used today", "Left today", "Exhausted until"], numeric=("Used today",))
        usage = read_usage(cfg.abspath(lc["usage_dir"]))
        if usage:
            df = pd.DataFrame(usage)
            df["day"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y-%m-%d")
            per = df.groupby(["day", "task"]).size().unstack(fill_value=0)
            fig = go.Figure([go.Bar(name=t, x=list(per.index), y=list(per[t])) for t in per.columns])
            fig.update_layout(barmode="stack")
            ui.apply_chart_style(fig, height=260)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        q.close()

    with ui.panel("admin-data"):
        ui.panel_header("Data freshness")
        m = bundle.meta
        ft, ex = m.get("fulltext") or {}, m.get("extract") or {}
        ui.stat_tiles([("Release built", str(m.get("generated_at", ""))[:10]), ("Papers", f"{len(bundle.df):,}"),
                       ("Full-text coverage", f"{float(ft.get('coverage') or 0):.0%}"),
                       ("Extracted (LLM)", (m.get("extraction_bundle") or {}).get("modes", {}).get("llm", 0)),
                       ("Waiting for LLM", ex.get("pending_llm", 0))])
