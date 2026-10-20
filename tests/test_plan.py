"""Research plans: context, verification, repair, LLM vs template generation, exports."""
from __future__ import annotations

import pytest
from conftest import make_config

from scholargrid.plan import (
    generate_plan,
    plan_context,
    plan_to_bibtex,
    plan_to_json,
    plan_to_markdown,
    repair_plan,
    supervisor_email,
    verify_plan,
)
from scholargrid.plan.generate import DATASET_RISK, parse_weeks
from scholargrid.plan.schemas import Choice, Milestone, ResearchPlan


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from scholargrid.artifacts import load_bundle
    from scholargrid.gaps.model import GapCard
    from scholargrid.pipeline import run

    cfg = make_config(tmp_path_factory.mktemp("plan"))
    run(cfg)
    b = load_bundle(cfg)
    card = GapCard.model_validate(next(c for cs in b.gaps["areas"].values() for c in cs))
    return cfg, b, card


def test_plan_context(built):
    cfg, b, card = built
    ctx = plan_context(card, b, cfg, b.embedder)
    refs = ctx["references"]
    ev_ids = list(dict.fromkeys(e.paper_id for e in card.evidence))
    assert [r["paper_id"] for r in refs[: len(ev_ids)]] == ev_ids
    assert [r["rid"] for r in refs] == [f"R{i + 1}" for i in range(len(refs))]
    related = [r for r in refs if r["role"] == "related"]
    assert related and not ({r["paper_id"] for r in related} & set(ev_ids))
    assert ctx["computed"]["n_related"] == sum(1 for r in refs if r["in_corpus"])
    assert ctx["computed"]["people"]


def _plan(**kw):
    base = dict(title="t", research_question="q", sub_questions=[], hypotheses=[], why_it_matters="w",
                contribution_type="empirical_study", method_sketch="m",
                baselines=[], datasets=[], metrics=[], experiments=[], minimum_viable_result="mvr", stretch_goals=[],
                milestones=[Milestone(weeks="1-4", goal="a", tasks=[], deliverable="d"),
                            Milestone(weeks="5-8", goal="b", tasks=[], deliverable="d")],
                risks=[], skills_to_learn=[], compute_estimate="c", questions_for_supervisor=[], references_used=[])
    base.update(kw)
    return ResearchPlan.model_validate(base)


def _ctx():
    return {"references": [{"rid": "R1", "title": "Paper one", "summary": "", "methods": ["LoRA"],
                            "datasets": ["SQuAD"], "metrics": ["F1"]}], "computed": {}}


def test_parse_weeks():
    assert parse_weeks("Week 1–2") == (1, 2) and parse_weeks("3") == (3, 3) and parse_weeks("x") is None


def test_verify_plan_catches_problems():
    ctx = _ctx()
    assert verify_plan(_plan(baselines=[Choice(name="LoRA", why="", ref="R1")]), ctx, 8) == []
    issues = verify_plan(_plan(baselines=[Choice(name="LoRA", why="", ref="R99")],
                               datasets=[Choice(name="ImageNet", why="", ref="R1")],
                               milestones=[Milestone(weeks="1-4", goal="a", tasks=[], deliverable="d"),
                                           Milestone(weeks="6-8", goal="b", tasks=[], deliverable="d")]), ctx, 8)
    text = " ".join(issues)
    assert "R99" in text and "ImageNet" in text and "missing [5]" in text


def test_repair_plan():
    ctx = _ctx()
    bad = _plan(datasets=[Choice(name="ImageNet", why="", ref="R1")],
                milestones=[Milestone(weeks="1-2", goal="a", tasks=[], deliverable="d"),
                            Milestone(weeks="6-8", goal="b", tasks=[], deliverable="d")])
    fixed = repair_plan(bad, ctx, verify_plan(bad, ctx, 8), 8)
    assert fixed.datasets == [] and any(DATASET_RISK in r.risk for r in fixed.risks)
    assert verify_plan(fixed, ctx, 8) == []


def test_generate_llm_and_template_and_exports(built):
    from scholargrid.llm import FakeLLM, NullLLM

    cfg, b, card = built
    doc = generate_plan(card, b, cfg, FakeLLM(cfg), b.embedder, weeks=10, background="Python", compute_access="laptop",
                        goal="exploring")
    assert doc.generated_by["mode"] == "llm"
    tpl = generate_plan(card, b, cfg, NullLLM(cfg), b.embedder, weeks=10, background="", compute_access="laptop",
                        goal="exploring")
    assert tpl.generated_by["mode"] == "template" and any("template" in w for w in tpl.warnings)
    assert verify_plan(tpl.plan, {"references": tpl.references}, 10) == []
    assert verify_plan(doc.plan, {"references": doc.references}, 10) == []      # repaired if needed
    for d in (doc, tpl):
        md = plan_to_markdown(d)
        assert all(m.weeks in md for m in d.plan.milestones)
        assert all(r["url"] in md for r in d.references if r.get("url"))
        assert "Computed from the referenced papers" in md
        assert plan_to_json(d).startswith("{") and "@misc" in plan_to_bibtex(d)
        assert len(supervisor_email(d).split()) <= 160
