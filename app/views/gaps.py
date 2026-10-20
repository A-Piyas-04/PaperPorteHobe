"""Open problems — evidence-backed Gap Cards for the active topic, plus cards from the wider area."""
from __future__ import annotations

import html

import streamlit as st

import ui
from scholargrid.gaps.model import TYPE_HELP, TYPE_LABELS

_SORTS = {"Score": lambda c: -float((c.get("scores") or {}).get("total", 0)),
          "Novelty": lambda c: -float((c.get("scores") or {}).get("novelty", 0)),
          "Feasibility": lambda c: -float((c.get("scores") or {}).get("feasibility", 0)),
          "Newest evidence": lambda c: -max([e.get("year") or 0 for e in c.get("evidence", [])] or [0])}
_LEVELS = ["low", "medium", "high"]


def toggle_saved(card: dict) -> None:
    ui.toggle_gap(card)


def make_plan(card: dict) -> None:
    st.session_state["plan_gap"] = card
    st.session_state["_goto"] = "plan"


def not_useful(card: dict, reason: str = "Other") -> None:
    from telemetry import track

    fb = dict(st.session_state.get("gap_feedback", {}))
    fb[card["id"]] = reason
    st.session_state.gap_feedback = fb
    track("gap_not_useful", gap_id=card["id"], type=card.get("type"), reason=reason)


def _recheck(card: dict) -> None:
    st.session_state["_recheck"] = card


def _extraction_missing(bundle) -> bool:
    """True when the served bundle carries no extracted problems at all.

    Every detector reads ``bundle.problems``; without it the engine can only ever return
    zero candidates, which is a build problem and not a property of the topic.
    """
    probs = getattr(bundle, "problems", None)
    return probs is None or not len(probs)


def _no_extraction_note() -> None:
    ui.note("<b>This bundle has no extracted open problems yet.</b> Gap finding reads author-stated "
            "limitations that are pulled out of paper full text by the extraction stage, and that stage "
            "has not run for the papers you are looking at — so no topic can produce cards yet. "
            "Run <code>python scripts/warm_topics.py</code> to do it for your saved topics, then restart "
            "the app.")


def _how_to_judge() -> None:
    from glossary import term

    with st.expander("How to judge a research gap"):
        st.markdown(
            f"- **Is it real?** Do several papers mention it? Read the quotes, not just the card. An "
            f"{term('open problem')} stated by one paper is weaker than one stated by five.\n"
            f"- **Is it still open?** Check the {term('novelty check')} line, and search the last 6 months yourself.\n"
            "- **Does it matter?** Who would use the result? Would a reviewer care?\n"
            f"- **Can you do it?** Is there public data, code for a {term('baseline')}, and enough compute and time?\n"
            "- **Would a supervisor agree?** Ask: is this known to be hard? Has someone just solved it? Is the scope "
            "right for me?", unsafe_allow_html=True)


def _next() -> None:
    chosen = bool(ui.saved_gap_ids() or st.session_state.get("plan_gap"))
    ui.next_step("Turn a problem you like into a starter plan", "plan",
                 "Use “Make a starter plan” on a card." if not chosen else "Plan the open problem you saved.",
                 enabled=chosen)


def _filters(cards: list[dict], key: str) -> list[dict]:
    types = sorted({c.get("type", "") for c in cards})
    c1, c2, c3, c4, c5 = st.columns([2, 1.1, 1.1, 1.2, 1.1])
    chosen = c1.multiselect("Type", types, format_func=lambda t: TYPE_LABELS.get(t, t), key=f"{key}_types",
                            placeholder="All types", help=" · ".join(f"{TYPE_LABELS[t]}: {TYPE_HELP[t]}" for t in types
                                                                       if t in TYPE_HELP))
    conf = c2.selectbox("Min. confidence", _LEVELS, key=f"{key}_conf")
    feas = c3.selectbox("Min. feasibility", _LEVELS, key=f"{key}_feas")
    beginner = c4.toggle("Beginner-friendly only", key=f"{key}_beg")
    sort = c5.selectbox("Sort by", list(_SORTS), key=f"{key}_sort")
    hidden = set((st.session_state.get("gap_feedback") or {}).keys())
    out = [c for c in cards
           if (not chosen or c.get("type") in chosen)
           and _LEVELS.index(c.get("confidence", "low")) >= _LEVELS.index(conf)
           and _LEVELS.index(c.get("feasibility", "medium")) >= _LEVELS.index(feas)
           and (not beginner or c.get("beginner_friendly"))
           and c.get("id") not in hidden]
    return sorted(out, key=_SORTS[sort])


def _render_cards(cards: list[dict], key: str, label: str | None = None) -> None:
    saved = ui.saved_gap_ids()
    pages = st.session_state.get("_pages") or {}
    for c in cards:
        with ui.panel(f"gapcard-{key}-{c['id']}"):
            ui.gap_card(c, key, on_recheck=_recheck, on_save=toggle_saved,
                        on_plan=make_plan if "plan" in pages else None, on_feedback=not_useful,
                        saved=c["id"] in saved, label=label)


def _by_area(bundle) -> None:
    areas = (bundle.gaps or {}).get("areas") or {}
    if not areas:
        ui.note("No area-level open problems in this release yet.")
        return
    names = {str(k): ui.area_name(bundle.clusters_meta, int(k)) for k in areas}
    cid = st.selectbox("Area", sorted(areas, key=lambda k: -len(areas[k])), format_func=lambda k: names.get(k, k),
                       key="gaps_area")
    _render_cards(_filters(areas.get(cid, []), "area"), "area", label="Area card")


def render() -> None:
    from scholargrid.gaps.workspace import gaps_for_workspace, load_cached_gaps, recheck_novelty

    if st.session_state.pop("_goto", None) == "plan":
        ui.goto("plan")
    bundle = ui.get_bundle()
    q = str(st.session_state.get("topic_query", "") or "").strip()
    if st.session_state.get("gaps_area_mode") or not q:
        ui.page_header("Open problems", "Open problems by research area",
                       "Cards built offline for each area of the collection. Read the evidence and treat every "
                       "card as a lead to verify.")
        if q:
            if st.button(f"← Back to your topic: {q}", type="tertiary", key="gaps_back_topic"):
                st.session_state["gaps_area_mode"] = False
                st.rerun()
        else:
            ui.note("<b>No topic yet.</b> Choose a topic to see its own open problems, or browse them by area below.")
            if st.button("Choose a topic", type="primary", key="gaps_need_topic"):
                ui.goto("start")
        _by_area(bundle)
        return
    ws = ui.require_workspace()
    if ws is None:
        return
    cfg = ui.get_config()
    ui.page_header("Open problems", f"Open problems in: {ws.query}",
                   "Each card is built from what authors wrote and what the papers contain. Read the evidence, "
                   "and treat every card as a lead to verify.")
    _how_to_judge()
    recheck = st.session_state.pop("_recheck", None)
    if recheck is not None:
        from limits import Busy, allow, heavy_slot, limit_message
        from telemetry import track

        if not allow("novelty_check"):
            ui.note(html.escape(limit_message("novelty_check")))
        else:
            try:
                with heavy_slot(), st.spinner("Checking recent papers…"):
                    new = recheck_novelty(recheck, ws, bundle, ui.get_searcher(ui.bundle_version()), cfg, ui.get_llm())
                st.session_state.pop("_gaps_data", None)
                track("novelty_checked", gap_id=recheck["id"], status=(new.get("novelty") or {}).get("status"))
            except Busy as exc:
                ui.note(html.escape(str(exc)))

    data = st.session_state.get("_gaps_data")
    if not data or data.get("ws_key") != ws.key:
        data = load_cached_gaps(ws.key, cfg)
        if data:
            st.session_state["_gaps_data"] = data
    if not data:
        from scholargrid.gaps.workspace import area_cards_for

        area_cards = area_cards_for(ws, bundle, cfg)
        if area_cards:
            with ui.panel("gaps-area"):
                ui.panel_header("From the wider area", f"{len(area_cards)} cards")
                _render_cards(area_cards, "warea", label="From the wider area")
        with ui.panel("gaps-generate"):
            ui.panel_header("Open problems for this topic")
            st.write("We look for problems several authors say are still open, untested combinations, narrow "
                     "evaluations, conflicting results and missing connections, then check recent papers.")
            if _extraction_missing(bundle):
                _no_extraction_note()
            if st.button("Find open problems for this topic (about 1–2 minutes)", type="primary", key="gaps_go"):
                import time

                from limits import Busy, allow, heavy_slot, limit_message
                from telemetry import track

                if not allow("gaps_generate"):
                    ui.note(html.escape(limit_message("gaps_generate")))
                    return
                t0 = time.time()
                try:
                    with heavy_slot(), st.status("Finding open problems…", expanded=True) as status:
                        bar = st.progress(0.0)

                        def progress(msg: str, frac: float) -> None:
                            status.update(label=msg)
                            bar.progress(min(1.0, max(0.0, float(frac))))

                        data = gaps_for_workspace(ws, bundle, ui.get_searcher(ui.bundle_version()), cfg,
                                                  ui.get_llm(), progress=progress)
                        status.update(label="Done", state="complete", expanded=False)
                except Busy as exc:
                    ui.note(html.escape(str(exc)))
                    return
                st.session_state["_gaps_data"] = data
                track("gaps_generated", n_cards=len(data.get("cards", [])), seconds=round(time.time() - t0, 1),
                      requests_used=(data.get("stats") or {}).get("requests_used", 0))
                st.rerun()
        _next()
        return

    stats = data.get("stats") or {}
    cards = data.get("cards") or []
    if stats.get("quota_stopped"):
        ui.note("<b>The free AI quota ran out while writing these cards.</b> Some use template wording; "
                "come back after the daily reset for full cards.")
    elif cards and all(not (c.get("generated_by") or {}).get("synthesized", True) for c in cards):
        ui.note("Cards use template wording (the AI writer is unavailable right now). The evidence is the same.")
    sub = st.session_state.get("gaps_subtopic")
    if sub is not None and ws.subtopic(int(sub)) is not None:
        name = ws.subtopic(int(sub)).name
        only = st.toggle(f"Only the sub-topic “{name}”", value=True, key="gaps_only_sub")
        if only:
            cards = [c for c in cards if str((c.get("scope") or {}).get("id", "")).endswith(f":{sub}")] or cards
    if not cards:
        if _extraction_missing(bundle):
            _no_extraction_note()
        elif not stats.get("candidates"):
            ui.note("No candidate gaps were found in the extracted problems for this topic. The papers here "
                    "may not be covered by the extraction stage yet — run "
                    "<code>python scripts/warm_topics.py</code> to extend it to your saved topics.")
        else:
            ui.note("This topic has few author-stated limitations in the analysed papers; try a broader topic or "
                    "another sub-topic.")
    else:
        _render_cards(_filters(cards, "ws"), "ws")
    area_cards = data.get("area_cards") or []
    if area_cards:
        with st.expander(f"From the wider area ({len(area_cards)})"):
            _render_cards(area_cards, "warea", label="From the wider area")
    already = data.get("already_explored") or []
    if already:
        with st.expander(f"Already explored ({len(already)})"):
            for c in already:
                st.markdown(f"- {html.escape(c.get('title', ''))}")
    st.caption(f"{stats.get('candidates', 0)} candidates checked · {stats.get('judged_out', 0)} judged implausible · "
               f"{stats.get('rejected_by_llm', 0)} rejected by the writer · {stats.get('already_addressed', 0)} "
               "already addressed by recent papers.")
    _next()
