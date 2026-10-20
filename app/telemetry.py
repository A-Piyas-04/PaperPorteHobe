"""Product analytics: anonymous, minimal, opt-out. Never blocks the UI.

Lives in the app (it needs session state and the cached repository); the library never imports ui.
Collected: event names, counts, timings and a hash of the topic text. Not collected: IP addresses, raw
topic text (unless telemetry.store_queries is on), paper text.
"""
from __future__ import annotations

import contextlib
import hashlib
import random
import secrets
import threading

import streamlit as st

EVENTS = {"app_open", "topic_started", "workspace_built", "reading_viewed", "gaps_generated",
          "gap_evidence_opened", "gap_saved", "gap_not_useful", "novelty_checked", "plan_generated",
          "plan_exported", "project_created", "project_returned", "paper_saved", "export_downloaded"}


def session_id() -> str:
    if "_sid" not in st.session_state:
        st.session_state["_sid"] = secrets.token_hex(16)
    return st.session_state["_sid"]


def query_props(cfg, query: str) -> dict:
    props = {"q_hash": hashlib.sha256(query.lower().encode()).hexdigest()[:16],
             "q_words": len(query.split())}
    if cfg["telemetry"]["store_queries"]:
        props["q"] = query[:300]
    return props


def track(name: str, **props) -> None:
    import ui

    try:
        cfg = ui.get_config()
        t = cfg["telemetry"]
        if not t["enabled"] or st.session_state.get("telemetry_opt_out") or name not in EVENTS:
            return
        if random.random() > float(t["sample_rate"]):
            return
        repo = ui.get_repo()
        if repo is None:
            return
        row = {"session": session_id(), "user_id": st.session_state.get("user_id"),
               "project_id": st.session_state.get("proj_id"), "name": name, "props": props}
        threading.Thread(target=_safe_insert, args=(repo, row), daemon=True).start()
    except Exception:
        return            # analytics must never break the app


def _safe_insert(repo, row: dict) -> None:
    with contextlib.suppress(Exception):          # analytics must never break the app
        repo.add_event(**row)
