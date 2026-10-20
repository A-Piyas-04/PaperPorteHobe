"""Plan — turn one Gap Card into a starter research plan, personalised to time, goal and compute."""
from __future__ import annotations

import hashlib
import html
import json

import streamlit as st

import ui

_GOALS = ["Course project", "Bachelor/master thesis", "First paper", "Just exploring"]
_COMPUTE = ["Laptop only", "One GPU", "Several GPUs", "Cloud credits"]


def _inputs_hash(gap_id: str, inputs: dict) -> str:
    return hashlib.sha1(json.dumps({"gap": gap_id, **inputs}, sort_keys=True).encode()).hexdigest()[:12]


def _choose_gap() -> dict | None:
    gap = st.session_state.get("plan_gap")
    if gap:
        return gap
    saved = ui.saved_gaps()
    ui.note("<b>Pick an open problem first.</b> Use “Make a starter plan” on a card in Open problems"
            + (", or choose one you saved below." if saved else "."))
    if saved:
        titles = {g["id"]: g.get("title", g["id"]) for g in saved}
        gid = st.selectbox("Saved open problems", list(titles), format_func=lambda k: titles[k], key="plan_pick")
        if st.button("Use this open problem", type="primary", key="plan_use"):
            st.session_state.plan_gap = next(g for g in saved if g["id"] == gid)
            st.rerun()
    if st.button("Go to Open problems", key="plan_to_gaps"):
        ui.goto("gaps")
    return None


def _summary(gap: dict) -> None:
    with ui.panel("plan-gap"):
        ui.panel_header("The open problem")
        st.markdown(f"<div class='sg-area-t'>{html.escape(gap.get('title', ''))}</div>"
                    f"<p class='sg-gap-statement'>{html.escape(gap.get('statement', ''))}</p>", unsafe_allow_html=True)
        text, _ = ui.novelty_line(gap.get("novelty") or {})
        st.caption(text)
        reasons = gap.get("why_it_might_not_be_a_gap") or []
        if reasons:
            st.markdown("<div class='sg-caveat'><b>Why this might not be a gap:</b> "
                        + html.escape("; ".join(reasons)) + "</div>", unsafe_allow_html=True)


def render_plan(doc) -> None:
    """Shared with My project (saved plans)."""
    _render_plan(doc)


def _render_plan(doc) -> None:
    from scholargrid.plan import plan_to_bibtex, plan_to_json, plan_to_markdown, supervisor_email

    p = doc.plan
    refs = {r["rid"]: r for r in doc.references}

    def ref(rid: str) -> str:
        r = refs.get(rid)
        return f"[{rid}]({r['url']})" if r and r.get("url") else rid

    for w in doc.warnings:
        ui.note(html.escape(w))
    tab_plan, tab_time, tab_res, tab_exp = st.tabs(["Plan", "Timeline", "Resources", "Export"])
    with tab_plan:
        st.markdown(f"### {p.title}")
        st.markdown(f"**Research question.** {p.research_question}")
        if p.sub_questions:
            st.markdown("**Sub-questions**\n" + "\n".join(f"- {x}" for x in p.sub_questions))
        if p.hypotheses:
            st.markdown("**Hypotheses**\n" + "\n".join(f"- {x}" for x in p.hypotheses))
        st.markdown(f"**Why it matters.** {p.why_it_matters}")
        st.markdown(f"**Method sketch.** {p.method_sketch}")
        for head, items in (("Baselines", p.baselines), ("Datasets", p.datasets)):
            st.markdown(f"**{head}**\n" + ("\n".join(f"- **{c.name}** — {c.why} ({ref(c.ref)})" for c in items)
                                          or "- none chosen yet"))
        if p.metrics:
            st.markdown("**Metrics:** " + ", ".join(p.metrics))
        if p.experiments:
            st.markdown("**Experiments**\n" + "\n".join(f"- {x}" for x in p.experiments))
        st.markdown(f"**Minimum viable result.** {p.minimum_viable_result}")
        if p.risks:
            st.markdown("**Risks**\n" + "\n".join(f"- {r.risk} — *{r.mitigation}*" for r in p.risks))
        if p.questions_for_supervisor:
            st.markdown("**Questions for a supervisor**\n" + "\n".join(f"- {q}" for q in p.questions_for_supervisor))
    with tab_time:
        st.markdown("<div class='sg-section'>Timeline</div>", unsafe_allow_html=True)
        for m in p.milestones:
            st.markdown(f"**Weeks {m.weeks} · {m.goal}**  \n" + "".join(f"- {t}\n" for t in m.tasks)
                        + f"\n*Deliverable:* {m.deliverable}")
        st.caption(f"Compute: {p.compute_estimate}")
    with tab_res:
        c = doc.computed
        reading = c.get("reading") or []
        if reading:
            st.markdown("<div class='sg-section'>Read first</div>", unsafe_allow_html=True)
            ui.reading_path_list(reading, key="plan")
        st.markdown("<div class='sg-section'>Computed from the referenced papers</div>", unsafe_allow_html=True)
        st.caption("Most frequent authors among the papers this plan builds on (not a ranking of researchers).")
        people = c.get("people") or []
        st.markdown("\n".join(f"- {('[' + x['name'] + '](' + x['url'] + ')') if x.get('url') else x['name']} "
                              f"({x['papers']} papers)" for x in people) or "- none found")
        venues = c.get("venues") or []
        st.markdown("**Venues:** " + (", ".join(f"{v} ({n})" for v, n in venues) or "no venue data"))
        links = c.get("code_links") or []
        st.markdown("**Code:**\n" + ("\n".join(f"- [{x['title']}]({x['code_url']})" for x in links) or "- none listed"))
        st.markdown("**References**\n" + "\n".join(f"- {r['rid']}: [{r['title']}]({r['url']}) ({r.get('year') or 'n.d.'})"
                                                  for r in doc.references))
    with tab_exp:
        st.markdown("<div class='sg-section'>Export</div>", unsafe_allow_html=True)
        from telemetry import track

        c1, c2, c3 = st.columns(3)
        for col, label, data, fname, mime, fmt in (
                (c1, "Markdown", plan_to_markdown(doc), "research_plan.md", "text/markdown", "md"),
                (c2, "JSON", plan_to_json(doc), "research_plan.json", "application/json", "json"),
                (c3, "BibTeX", plan_to_bibtex(doc), "research_plan.bib", "application/x-bibtex", "bib")):
            col.download_button(label, data, fname, mime, use_container_width=True, key=f"plan_dl_{fmt}_{doc.id}",
                                on_click=track, args=("plan_exported",), kwargs={"format": fmt})
        st.text_area("Email draft to a possible supervisor", supervisor_email(doc), height=260,
                     key=f"plan_email_{doc.id}")


def render() -> None:
    from scholargrid.plan import PlanDocument, generate_plan

    ui.journey_bar("plan")
    gap = _choose_gap()
    if gap is None:
        return
    cfg = ui.get_config()
    pc = cfg["plan"]
    ui.page_header("Plan", "Your starter research plan",
                   "A realistic first project for this open problem, grounded in the papers behind it.")
    _summary(gap)
    with st.expander("What makes a good first project"):
        st.markdown(
            "- **Small and concrete.** One question you can answer in the time you have.\n"
            "- **Start by reproducing a baseline.** It teaches you the code, the data and the evaluation, and "
            "gives you a number to beat.\n"
            "- **One clear change.** Change one thing and measure it; resist building a whole new system.\n"
            "- **A defined evaluation.** Decide the dataset and metric before you start.\n"
            "- **A minimum viable result** you could write up even if the idea doesn't work.")
    with st.form("plan-inputs"):
        options = list(pc["weeks_options"])
        default = pc["default_weeks"] if pc["default_weeks"] in options else options[len(options) // 2]
        weeks = st.select_slider("Weeks available", options=options, value=default, key="plan_weeks")
        goal = st.radio("Goal", _GOALS, horizontal=True, key="plan_goal")
        compute = st.selectbox("Compute you can use", _COMPUTE, key="plan_compute")
        background = st.text_area("Your background (optional)", max_chars=int(pc["max_background_chars"]),
                                  placeholder="e.g. comfortable with Python and PyTorch, took an NLP course, no "
                                              "research experience yet", key="plan_background")
        st.caption("This text is sent to Google Gemini (free tier). Don't include personal or confidential "
                   "information.")
        submitted = st.form_submit_button("Create my starter plan", type="primary")
    inputs = {"weeks": int(weeks), "goal": goal, "compute_access": compute, "background": background.strip()}
    key = _inputs_hash(gap["id"], inputs)
    plans = st.session_state.setdefault("plans", {})
    attempt = int(st.session_state.get("plan_attempt", 0))
    if submitted or st.session_state.pop("_plan_regen", False):
        ws = None
        if st.session_state.get("topic_query"):
            try:
                ws = ui.get_workspace(str(st.session_state["topic_query"]))
            except Exception:
                ws = None
        import time

        from telemetry import track

        t0 = time.time()
        with st.status("Writing your plan… (about a minute)", expanded=False) as status:
            bundle = ui.get_bundle()
            doc = generate_plan(gap, bundle, cfg, ui.get_llm(), bundle.embedder, weeks=inputs["weeks"],
                                background=inputs["background"], compute_access=compute, goal=goal, workspace=ws,
                                attempt=attempt)
            status.update(label="Done", state="complete")
        track("plan_generated", mode=doc.generated_by.get("mode"), seconds=round(time.time() - t0, 1),
              weeks=inputs["weeks"])
        plans.setdefault(gap["id"], {})[key] = doc.model_dump(mode="json")
        st.session_state["plan_last"] = key
    stored = (plans.get(gap["id"]) or {}).get(key) or (plans.get(gap["id"]) or {}).get(
        st.session_state.get("plan_last", ""))
    if not stored:
        return
    doc = PlanDocument.model_validate(stored)
    _render_plan(doc)
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Regenerate", key="plan_regen"):
        st.session_state["plan_attempt"] = attempt + 1
        st.session_state["_plan_regen"] = True
        st.rerun()
    persistent = ui.get_repo() is not None
    if c2.button("Save plan to project" if persistent else "Save plan to this session", key="plan_save"):
        try:
            durable = ui.save_plan(doc.model_dump(mode="json"))
            st.toast("Plan saved to your project." if durable else "Plan saved for this session.")
        except ValueError as exc:
            ui.note(f"<b>Couldn't save.</b> {html.escape(str(exc))}")
    ui.project_notices()
    st.session_state.j_plan = True
    ui.next_step("Save it to your project and share it with a supervisor", "project",
                 "Use “Save plan to project” above, then open My project.")
