"""Who is the current user and which project is active (session + URL).

Anonymous projects live behind a capability link ``?p=<project_id>&k=<token>``; the token is a secret, so
it is never logged. With ``persistence.auth: oidc`` users sign in with Streamlit's ``st.login``.
"""
from __future__ import annotations

import time

import streamlit as st

from scholargrid.persistence.repo import Access


def current_user(cfg, repo) -> dict | None:
    if cfg["persistence"]["auth"] != "oidc" or repo is None:
        return None
    user = getattr(st, "user", None)
    try:
        if user is None or not user.is_logged_in:
            return None
        email = user.email
    except Exception:          # auth not configured in secrets.toml
        return None
    key = f"_user_{email}"
    if key not in st.session_state:
        st.session_state[key] = repo.upsert_user(email=email, name=getattr(user, "name", "") or "")
    return st.session_state[key]


def active_access(cfg, repo) -> Access:
    """Signed-in user, else a capability link from the URL (?p=&k=) or the session."""
    qp = st.query_params
    if "p" in qp and "k" in qp and (qp["p"] != st.session_state.get("proj_id")
                                    or qp["k"] != st.session_state.get("proj_token")):
        st.session_state["proj_id"], st.session_state["proj_token"] = qp["p"], qp["k"]
        st.session_state["_proj_rev"] = st.session_state.get("_proj_rev", 0) + 1
    user = current_user(cfg, repo)
    return Access(user_id=user["id"] if user else None,
                  project_id=st.session_state.get("proj_id"), token=st.session_state.get("proj_token"))


def _rate_limit_creation(cfg) -> None:
    now = time.time()
    recent = [t for t in st.session_state.get("_proj_created", []) if now - t < 3600]
    if len(recent) >= int(cfg["persistence"]["create_rate_per_hour"]):
        raise ValueError("Too many new projects from this browser in the last hour. Try again later.")
    st.session_state["_proj_created"] = [*recent, now]


def _migrate_session_items(repo, pid: str, access: Access) -> None:
    """Move the old session reading list and saved gaps into the new project, once."""
    for p in list((st.session_state.pop("saved_papers", None) or {}).values()):
        try:
            repo.add_paper(pid, access, p, source="session")
        except ValueError:
            break
    for g in list((st.session_state.pop("saved_gaps", None) or {}).values()):
        try:
            repo.add_gap(pid, access, g)
        except ValueError:
            break


def ensure_project(cfg, repo, *, title: str, topic: str, bundle_version: str = "") -> str:
    """Active project id; creates one on first save. Anonymous: puts ?p=&k= into the URL."""
    acc = active_access(cfg, repo)
    pid = st.session_state.get("proj_id")
    if pid:
        return pid
    _rate_limit_creation(cfg)
    pid, token = repo.create_project(acc, title=title or "My research project", topic_query=topic,
                                     scope_name=str(cfg["scope"]["name"]), bundle_version=bundle_version)
    st.session_state["proj_id"], st.session_state["proj_token"] = pid, token
    if not acc.user_id:
        st.query_params["p"], st.query_params["k"] = pid, token
        st.session_state["show_link_notice"] = True
    _migrate_session_items(repo, pid, Access(acc.user_id, pid, token))
    return pid


def project_link() -> str:
    """Full share URL if Streamlit knows it, else the relative ?p=&k= part."""
    pid, token = st.session_state.get("proj_id"), st.session_state.get("proj_token")
    if not pid or not token:
        return ""
    rel = f"?p={pid}&k={token}"
    base = ""
    try:
        base = str(getattr(st.context, "url", "") or "").split("?", 1)[0]
    except Exception:
        base = ""
    return (base + rel) if base else rel
