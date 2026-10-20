"""Evaluate — blind rating of frozen rounds (gap cards, extractions, search, reading paths, plans).

Raters never see the system's own scores, confidence or novelty verdicts while rating: they anchor
judgments. Each answer is saved immediately as a Rating row.
"""
from __future__ import annotations

import hashlib
import html
import os

import streamlit as st

import ui

_RUBRIC_GAP = {
    "real": ("Real research problem?", ["yes", "partly", "no"]),
    "open": ("Still open, to your knowledge?", ["yes", "unsure", "no"]),
    "specific": ("Specific enough to start?", ["1 vague", "2 workable", "3 concrete"]),
    "feasible": ("Feasible for a master's student in ~3 months?", ["yes", "maybe", "no"]),
    "supported": ("Does the evidence support the claim?", ["yes", "partly", "no"]),
    "suggest": ("Would you suggest it to a student?", ["yes", "no"]),
}


def _radio(label: str, options: list[str], key: str) -> str:
    v = st.radio(label, options, index=None, horizontal=True, key=key)
    return str(v).split(" ")[0] if v is not None else ""


def _gap_item(p: dict, key: str) -> dict | None:
    st.markdown(f"### {html.escape(p.get('title', ''))}")
    st.markdown(f"<p class='sg-gap-statement'>{html.escape(p.get('statement', ''))}</p>", unsafe_allow_html=True)
    if p.get("why_it_matters"):
        st.markdown(f"**Why it matters.** {p['why_it_matters']}")
    st.markdown("**Evidence**")
    for e in p.get("evidence", []):
        line = f"- [{e.get('title', '')}]({e.get('url', '')}) ({e.get('year') or 'n.d.'})"
        if e.get("quote"):
            line += f"\n  > “{e['quote']}”" + (f" — {e['section']}" if e.get("section") else "")
        elif e.get("note"):
            line += f" — {e['note']}"
        st.markdown(line)
    if p.get("first_steps"):
        st.markdown("**First steps**\n" + "\n".join(f"- {s}" for s in p["first_steps"]))
    if p.get("why_it_might_not_be_a_gap"):
        st.markdown("**Why it might not be a gap**\n" + "\n".join(f"- {s}" for s in p["why_it_might_not_be_a_gap"]))
    with st.form(f"rate-{key}"):
        ans = {k: _radio(label, opts, f"{key}_{k}") for k, (label, opts) in _RUBRIC_GAP.items()}
        ans["solved_by"] = st.text_input("If not open: link to a paper that solves it (optional)", key=f"{key}_solved")
        comment = st.text_area("Comment (optional)", key=f"{key}_comment")
        if st.form_submit_button("Save rating", type="primary"):
            if any(not ans[k] for k in _RUBRIC_GAP):
                st.warning("Please answer every question (the comment and link are optional).")
                return None
            return {"criteria": ans, "comment": comment}
    return None


def _problem_item(p: dict, key: str) -> dict | None:
    st.markdown(f"### [{html.escape(p.get('title', ''))}]({p.get('url', '')})")
    st.caption(f"Section: {p.get('section') or 'unknown'}")
    st.markdown(f"> “{p.get('quote', '')}”")
    st.markdown(f"**Statement:** {p.get('statement', '')}")
    with st.form(f"rate-{key}"):
        stated = _radio("Stated by the authors as open or a limitation?", ["yes", "no"], f"{key}_stated")
        faithful = _radio("Statement faithful to the quote?", ["yes", "partly", "no"], f"{key}_faithful")
        spec = _radio("Specificity", ["1 vague", "2 names the issue", "3 concrete"], f"{key}_spec")
        comment = st.text_area("Comment (optional)", key=f"{key}_comment")
        if st.form_submit_button("Save rating", type="primary"):
            if not (stated and faithful and spec):
                st.warning("Please answer every question.")
                return None
            return {"criteria": {"stated": stated, "faithful": faithful, "specificity": int(spec)}, "comment": comment}
    return None


def _facets_item(p: dict, key: str) -> dict | None:
    st.markdown(f"### [{html.escape(p.get('title', ''))}]({p.get('url', '')})")
    st.write(p.get("abstract", ""))
    with st.form(f"rate-{key}"):
        st.caption("Untick anything that is wrong. Type what is missing (comma-separated).")
        wrong: list[str] = []
        missing: list[str] = []
        shown = 0
        for field in ("tasks", "methods", "datasets", "metrics"):
            vals = p.get(field, [])
            st.markdown(f"**{field.title()}**")
            for v in vals:
                shown += 1
                if not st.checkbox(v, value=True, key=f"{key}_{field}_{v}"):
                    wrong.append(f"{field}:{v}")
            extra = st.text_input(f"Missing {field}", key=f"{key}_{field}_missing")
            missing += [f"{field}:{x.strip()}" for x in extra.split(",") if x.strip()]
        comment = st.text_area("Comment (optional)", key=f"{key}_comment")
        if st.form_submit_button("Save rating", type="primary"):
            return {"criteria": {"n_shown": shown, "n_wrong": len(wrong), "n_missing": len(missing), "wrong": wrong,
                                 "missing": missing}, "comment": comment}
    return None


def _search_item(p: dict, key: str) -> dict | None:
    st.markdown(f"**Query:** {html.escape(p.get('query', ''))}")
    st.markdown(f"### [{html.escape(p.get('title', ''))}]({p.get('url', '')})")
    st.write(p.get("abstract", ""))
    with st.form(f"rate-{key}"):
        g = st.radio("Relevance", ["0 not relevant", "1 partly relevant", "2 relevant"], index=None, horizontal=True,
                     key=f"{key}_grade")
        if st.form_submit_button("Save rating", type="primary"):
            if g is None:
                st.warning("Please choose a grade.")
                return None
            return {"criteria": {"grade": int(str(g)[0]), "rank": p.get("rank")}, "comment": ""}
    return None


def _plan_item(p: dict, key: str) -> dict | None:
    from scholargrid.plan import PlanDocument, plan_to_markdown

    st.markdown(plan_to_markdown(PlanDocument.model_validate(p)))
    with st.form(f"rate-{key}"):
        scores = {f: st.select_slider(label, options=[1, 2, 3, 4, 5], value=3, key=f"{key}_{f}")
                  for f, label in (("realistic", "Realistic scope"), ("grounded", "Grounded in the references"),
                                   ("useful", "First steps useful"))}
        errors = st.text_area("Factual errors (optional)", key=f"{key}_errors")
        if st.form_submit_button("Save rating", type="primary"):
            return {"criteria": {**scores, "errors": errors}, "comment": ""}
    return None


def _reading_item(p: dict, key: str) -> dict | None:
    st.markdown(f"### Reading path: {html.escape(str(p.get('label', '')))}")
    for i, it in enumerate(p.get("path", []), start=1):
        st.markdown(f"{i}. [{it.get('title', '')}]({it.get('url', '')}) ({it.get('year') or 'n.d.'}) — "
                    f"*{it.get('step', '')}*: {it.get('reason', '')}")
    with st.form(f"rate-{key}"):
        right = st.select_slider("Right papers?", options=[1, 2, 3, 4, 5], value=3, key=f"{key}_right")
        order = st.select_slider("Sensible order?", options=[1, 2, 3, 4, 5], value=3, key=f"{key}_order")
        missing = st.text_area("Anything essential missing? (optional)", key=f"{key}_missing")
        if st.form_submit_button("Save rating", type="primary"):
            return {"criteria": {"right_papers": right, "order": order, "missing": missing}, "comment": ""}
    return None


_RENDER = {"gap_card": _gap_item, "problem": _problem_item, "facets": _facets_item, "search": _search_item,
           "plan": _plan_item, "reading": _reading_item}


def _save_search_judgment(cfg, payload: dict, grade: int) -> None:
    from scholargrid.utils import load_json, save_json

    target = cfg.abspath(cfg["validation"]["judgments_file"] or "configs/search_judgments.json")
    data = load_json(target) if os.path.exists(target) else {"judgments": {}}
    data.setdefault("judgments", {}).setdefault(payload["query"], {})[str(payload["arxiv_id"])] = int(grade)
    save_json(data, target)


def render() -> None:
    from scholargrid.evaluation import list_rounds, load_round, rater_queue

    if not ui.eval_allowed():
        ui.note("This page is only available in evaluation mode.")
        return
    cfg = ui.get_config()
    repo = ui.get_repo()
    ui.page_header("Evaluation", "Rate items", "Blind ratings: the system's own judgments are hidden while you rate.")
    if repo is None:
        ui.note("Ratings need the projects database (persistence.enabled).")
        return
    rater = st.text_input("Your rater code", value=st.session_state.get("rater") or ui.signed_in_email(),
                          placeholder="e.g. R1-mahin", key="rater_input").strip()
    if not rater:
        st.info("Enter a rater code to start.")
        return
    st.session_state["rater"] = rater
    rounds = list_rounds(cfg)
    if not rounds:
        ui.note("No rounds yet. Create one with <code>python -m scholargrid.evaluation round …</code>.")
        return
    name = st.selectbox("Round", rounds, key="eval_round")
    rnd = load_round(cfg, name)
    rated_by: dict = {}
    mine = 0
    for r in repo.ratings(batch=name):
        rated_by.setdefault(r["item_id"], set()).add(r["rater"])
        mine += int(r["rater"] == rater)
    queue = rater_queue(rnd, rater, rated_by)
    pos_key = f"eval_pos_{name}_{rater}"
    pos = min(int(st.session_state.get(pos_key, 0)), max(0, len(queue) - 1))
    st.progress(mine / max(1, mine + len(queue)), text=f"{mine} of {mine + len(queue)} rated")
    if not queue:
        st.success("All done for this round. Thank you!")
        return
    item = queue[pos]
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("← Back", key="eval_back", disabled=pos == 0):
        st.session_state[pos_key] = pos - 1
        st.rerun()
    if c2.button("Skip →", key="eval_skip", disabled=pos >= len(queue) - 1):
        st.session_state[pos_key] = pos + 1
        st.rerun()
    st.caption(f"Item {pos + 1} of {len(queue)} left · {item['item_type'].replace('_', ' ')}")
    with ui.panel("eval-item"):
        key = "e_" + hashlib.sha1(item["item_id"].encode("utf-8")).hexdigest()[:12]
        answer = _RENDER[item["item_type"]](item["payload"], key)
    if answer is not None:
        repo.add_rating(rater=rater, item_type=item["item_type"], item_id=item["item_id"], batch=name,
                        context=item.get("context", {}), criteria=answer["criteria"], comment=answer.get("comment", ""))
        if item["item_type"] == "search":
            _save_search_judgment(cfg, item["payload"], int(answer["criteria"]["grade"]))
        st.session_state[pos_key] = pos
        st.toast("Saved.")
        st.rerun()
