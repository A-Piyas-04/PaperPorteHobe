"""My project — saved papers, open problems, plans and notes that stay after the tab is closed."""
from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

import ui
from scholargrid.bibtex import to_bibtex
from scholargrid.persistence.repo import GAP_STATUSES, PAPER_STATUSES, Access, AccessDenied

_PAPER_LABELS = {"to_read": "To read", "reading": "Reading", "read": "Read", "key": "Key paper"}
_GAP_LABELS = {"considering": "Considering", "shortlisted": "Shortlisted", "chosen": "Chosen", "rejected": "Rejected"}


def _session_fallback() -> None:
    from views import library

    ui.note("Projects are stored for this session only on this site (persistence is turned off).")
    library.render()
    gaps = ui.saved_gaps()
    if gaps:
        st.markdown("<div class='sg-section'>Saved open problems</div>", unsafe_allow_html=True)
        for g in gaps:
            st.markdown(f"- {html.escape(g.get('title', ''))}")


def _empty() -> None:
    ui.page_header("My project", "Nothing saved yet")
    ui.empty_state("Your project is empty",
                   "Save papers, open problems and plans; they'll appear here and stay after you close the tab.",
                   [("Start with a topic", "start"), ("Search papers", "home")])


def _header(repo, pid: str, acc: Access, info: dict, user: dict | None) -> None:
    ui.page_header("My project", info["title"],
                   (f"Topic: {info['topic_query']} · " if info["topic_query"] else "")
                   + f"created {info['created_at'][:10]} · updated {info['updated_at'][:10]}")
    c = info["counts"]
    ui.stat_tiles([("Papers", c["papers"]), ("Open problems", c["gaps"]), ("Plans", c["plans"]), ("Notes", c["notes"])])
    with st.expander("Rename or switch project"):
        title = st.text_input("Project title", value=info["title"], max_chars=200, key="proj_title")
        if st.button("Rename", key="proj_rename") and title.strip() and title != info["title"]:
            repo.rename(pid, acc, title)
            ui._bump()
            st.rerun()
        if user:
            projects = repo.list_projects(user["id"])
            if projects:
                ids = [p["id"] for p in projects]
                names = {p["id"]: p["title"] for p in projects}
                choice = st.selectbox("Your projects", ids, index=ids.index(pid) if pid in ids else 0,
                                      format_func=lambda k: names[k], key="proj_switch")
                if choice != pid and st.button("Open", key="proj_open"):
                    st.session_state["proj_id"], st.session_state["proj_token"] = choice, None
                    st.query_params.clear()
                    ui._bump()
                    st.rerun()
            if st.button("New project", key="proj_new"):
                new_id, _ = repo.create_project(Access(user_id=user["id"]), title="My research project")
                st.session_state["proj_id"], st.session_state["proj_token"] = new_id, None
                ui._bump()
                st.rerun()


def _papers_tab(repo, pid: str, acc: Access) -> None:
    papers = repo.papers(pid, acc)
    if not papers:
        st.info("No saved papers yet. Use “Save to project” on any paper.")
        return
    f1, f2 = st.columns(2)
    status = f1.multiselect("Status", list(PAPER_STATUSES), format_func=lambda s: _PAPER_LABELS[s], key="pf_status")
    tags = sorted({t for p in papers for t in p["tags"]})
    tag = f2.multiselect("Tags", tags, key="pf_tags")
    shown = [p for p in papers if (not status or p["status"] in status) and (not tag or set(tag) & set(p["tags"]))]
    frame = pd.DataFrame([{"remove": False, "title": p["title"], "link": p["url"], "year": p["year"],
                           "status": p["status"], "tags": ", ".join(p["tags"]), "note": p["note"],
                           "paper_id": p["paper_id"]} for p in shown])
    edited = st.data_editor(
        frame, hide_index=True, use_container_width=True, key="proj_papers_editor",
        disabled=["title", "link", "year", "paper_id"],
        column_config={"remove": st.column_config.CheckboxColumn("Remove", width="small"),
                       "link": st.column_config.LinkColumn("Link", display_text="open"),
                       "status": st.column_config.SelectboxColumn("Status", options=list(PAPER_STATUSES)),
                       "tags": st.column_config.TextColumn("Tags (comma-separated)"),
                       "note": st.column_config.TextColumn("Note"), "paper_id": None})
    c1, c2, c3 = st.columns(3)
    if c1.button("Apply changes", type="primary", key="proj_apply"):
        errors = []
        by_id = {p["paper_id"]: p for p in shown}
        for row in edited.to_dict("records"):
            old = by_id.get(row["paper_id"])
            if old is None:
                continue
            try:
                if row["remove"]:
                    repo.remove_paper(pid, acc, row["paper_id"])
                    continue
                new_tags = [t.strip() for t in str(row["tags"] or "").split(",") if t.strip()]
                if row["status"] != old["status"] or new_tags != old["tags"] or (row["note"] or "") != old["note"]:
                    repo.update_paper(pid, acc, row["paper_id"], status=row["status"], tags=new_tags,
                                      note=row["note"] or "")
            except ValueError as exc:
                errors.append(f"{old['title'][:40]}: {exc}")
        ui._bump()
        if errors:
            ui.note("<b>Some changes were not saved.</b> " + html.escape("; ".join(errors)))
        else:
            st.rerun()
    rows = [{**p["payload"], "paper_id": p["paper_id"], "title": p["title"], "url": p["url"],
             "date": p["payload"].get("date") or (str(p["year"]) if p["year"] else "")} for p in shown]
    c2.download_button("BibTeX", to_bibtex(rows), "project_papers.bib", "application/x-bibtex", use_container_width=True)
    c3.download_button("CSV", pd.DataFrame(rows).to_csv(index=False), "project_papers.csv", "text/csv",
                       use_container_width=True)


def _make_plan(card: dict) -> None:
    st.session_state["plan_gap"] = card
    st.session_state["_proj_goto"] = "plan"


def _gaps_tab(repo, pid: str, acc: Access) -> None:
    gaps = repo.gaps(pid, acc)
    if not gaps:
        st.info("No saved open problems yet. Use “Save to project” on a card in Open problems.")
        return
    for g in gaps:
        card = g["payload"]
        with ui.panel(f"proj-gap-{g['gap_id']}"):
            ui.gap_card(card, "proj", on_plan=_make_plan)
            c1, c2 = st.columns([1, 2])
            status = c1.selectbox("Status", list(GAP_STATUSES), index=list(GAP_STATUSES).index(g["status"]),
                                  format_func=lambda s: _GAP_LABELS[s], key=f"gs_{g['gap_id']}")
            note = c2.text_area("Your note", value=g["note"], key=f"gn_{g['gap_id']}", height=80)
            b1, b2, _ = st.columns([1, 1, 3])
            if b1.button("Save", key=f"gsave_{g['gap_id']}"):
                try:
                    repo.update_gap(pid, acc, g["gap_id"], status=status, note=note)
                    ui._bump()
                    st.rerun()
                except ValueError as exc:
                    ui.note(html.escape(str(exc)))
            if b2.button("Remove", key=f"gdel_{g['gap_id']}"):
                repo.remove_gap(pid, acc, g["gap_id"])
                ui._bump()
                st.rerun()
    if len(gaps) >= 2:
        st.markdown("<div class='sg-section'>Compare</div>", unsafe_allow_html=True)
        names = {g["gap_id"]: g["payload"].get("title", g["gap_id"]) for g in gaps}
        pick = st.multiselect("Pick 2–3 open problems", list(names), format_func=lambda k: names[k],
                              max_selections=3, key="proj_compare")
        if len(pick) >= 2:
            cols = st.columns(len(pick))
            by_id = {g["gap_id"]: g for g in gaps}
            for col, gid in zip(cols, pick):
                g = by_id[gid]
                c = g["payload"]
                s = c.get("scores") or {}
                with col:
                    st.markdown(f"**{html.escape(c.get('title', ''))}**")
                    st.caption(c.get("statement", ""))
                    st.markdown(f"- Evidence: {len({e.get('paper_id') for e in c.get('evidence', [])})} papers\n"
                                f"- Novelty: {str((c.get('novelty') or {}).get('status', 'not_checked')).replace('_', ' ')}\n"
                                f"- Feasibility: {c.get('feasibility', '?')}\n"
                                f"- Score: {float(s.get('total', 0) or 0):.2f}\n"
                                f"- Status: {_GAP_LABELS.get(g['status'], g['status'])}")
                    if g["note"]:
                        st.text(g["note"])


def _plans_tab(repo, pid: str, acc: Access) -> None:
    from scholargrid.plan import PlanDocument, plan_to_markdown

    plans = repo.plans(pid, acc)
    if not plans:
        st.info("No saved plans yet. Make a starter plan from an open problem and save it.")
        return
    labels = {p["id"]: f"{p['label']} ({p['created_at'][:10]})" for p in plans}
    pick = st.selectbox("Saved plans", list(labels), format_func=lambda k: labels[k], key="proj_plan_pick")
    plan = next(p for p in plans if p["id"] == pick)
    doc = PlanDocument.model_validate(plan["payload"])
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Markdown", plan_to_markdown(doc), "research_plan.md", "text/markdown",
                       use_container_width=True, key=f"proj_plan_md_{pick}")
    if c2.button("Delete plan", key=f"proj_plan_del_{pick}"):
        repo.remove_plan(pid, acc, int(pick))
        ui._bump()
        st.rerun()
    from views.plan import render_plan

    render_plan(doc)


def _notes_tab(repo, pid: str, acc: Access) -> None:
    body = st.text_area("New note", key="proj_new_note", height=100, placeholder="What did you learn today? What's next?")
    if st.button("Add", key="proj_add_note") and body.strip():
        try:
            repo.add_note(pid, acc, body)
            ui._bump()
            st.rerun()
        except ValueError as exc:
            ui.note(html.escape(str(exc)))
    for n in repo.notes(pid, acc):
        with st.container(key=f"note-{n['id']}"):
            st.caption(n["created_at"][:16].replace("T", " "))
            st.text(n["body"])
            with st.expander("Edit"):
                new = st.text_area("Note", value=n["body"], key=f"note_edit_{n['id']}")
                e1, e2, _ = st.columns([1, 1, 3])
                if e1.button("Save", key=f"note_save_{n['id']}"):
                    repo.update_note(pid, acc, n["id"], new)
                    st.rerun()
                if e2.button("Delete", key=f"note_del_{n['id']}"):
                    repo.remove_note(pid, acc, n["id"])
                    ui._bump()
                    st.rerun()


def _settings_tab(repo, pid: str, acc: Access, info: dict, user: dict | None) -> None:
    from project_access import project_link

    st.markdown("<div class='sg-section'>Share link</div>", unsafe_allow_html=True)
    if acc.token or (user and info.get("owner_id") == user["id"]):
        if not acc.token:
            st.session_state["proj_token"] = repo.share_token(pid, acc)
        st.caption("Anyone with this link can view and edit the project.")
        st.code(project_link(), language=None)
        if st.button("Create a new link (the old one stops working)", key="proj_rotate"):
            token = repo.rotate_token(pid, acc)
            st.session_state["proj_token"] = token
            st.query_params["p"], st.query_params["k"] = pid, token
            st.rerun()
    st.markdown("<div class='sg-section'>Export and import</div>", unsafe_allow_html=True)
    st.download_button("Export project (JSON)", json.dumps(repo.export_project(pid, acc), indent=2, ensure_ascii=False),
                       "scholargrid_project.json", "application/json", key="proj_export")
    up = st.file_uploader("Import a project file (JSON, up to 2 MB)", type=["json"], key="proj_import")
    if up is not None and st.button("Import as a new project", key="proj_import_go"):
        try:
            if up.size > 2 * 1024 * 1024:
                raise ValueError("The file is larger than 2 MB.")
            new_id, token = repo.import_project(json.loads(up.getvalue().decode("utf-8")),
                                                Access(user_id=user["id"] if user else None))
            st.session_state["proj_id"], st.session_state["proj_token"] = new_id, token
            if not user:
                st.query_params["p"], st.query_params["k"] = new_id, token
            ui._bump()
            st.rerun()
        except (ValueError, UnicodeDecodeError) as exc:
            ui.note(f"<b>Import failed.</b> {html.escape(str(exc))}")
    if user and acc.token and not info.get("owned") and st.button("Add this project to my account", key="proj_claim"):
        repo.claim(pid, acc.token, user["id"])
        ui._bump()
        st.rerun()
    st.markdown("<div class='sg-section'>Delete project</div>", unsafe_allow_html=True)
    confirm = st.text_input("Type the project title to confirm", key="proj_delete_confirm")
    if st.button("Delete project permanently", key="proj_delete", disabled=confirm.strip() != info["title"]):
        repo.delete_project(pid, acc)
        for k in ("proj_id", "proj_token"):
            st.session_state.pop(k, None)
        st.query_params.clear()
        ui._bump()
        st.rerun()


def render() -> None:
    from project_access import active_access, current_user

    if st.session_state.pop("_proj_goto", None) == "plan":
        ui.goto("plan")
    ui.journey_bar("project")
    repo = ui.get_repo()
    if repo is None:
        _session_fallback()
        return
    cfg = ui.get_config()
    acc = active_access(cfg, repo)
    user = current_user(cfg, repo)
    if not acc.project_id and user:
        projects = repo.list_projects(user["id"])
        if projects:
            st.session_state["proj_id"] = projects[0]["id"]
            acc = active_access(cfg, repo)
    if not acc.project_id:
        _empty()
        return
    try:
        info = repo.get_project(acc.project_id, acc)
    except AccessDenied:
        ui.note("<b>This project link doesn't work.</b> It may have been replaced by a new link or deleted.")
        for k in ("proj_id", "proj_token"):
            st.session_state.pop(k, None)
        return
    pid = acc.project_id
    ui.project_notices()
    _header(repo, pid, acc, info, user)
    tabs = st.tabs(["Papers", "Open problems", "Plans", "Notes", "Settings"])
    with tabs[0]:
        _papers_tab(repo, pid, acc)
    with tabs[1]:
        _gaps_tab(repo, pid, acc)
    with tabs[2]:
        _plans_tab(repo, pid, acc)
    with tabs[3]:
        _notes_tab(repo, pid, acc)
    with tabs[4]:
        _settings_tab(repo, pid, acc, info, user)
    if info["counts"]["gaps"] or info["counts"]["plans"]:
        st.session_state.j_project = True
    ui.next_step("Explore another topic", "start")
