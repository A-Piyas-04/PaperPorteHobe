"""Guards between the LLM and the user, and the synthesize retry / template path."""
from __future__ import annotations

from conftest import make_config

from scholargrid.gaps.model import Evidence, GapCandidate
from scholargrid.gaps.prompts import GapDraft
from scholargrid.gaps.synthesize import synthesize
from scholargrid.gaps.verify import foreign_entities, verify_text
from scholargrid.llm import FakeLLM


def _cand():
    ev = [Evidence(eid="E1", paper_id="p1", title="Robust QA on SQuAD", role="states_problem",
                   quote="Our model fails on long SQuAD contexts.")]
    return GapCandidate(type="stated_open_problem", scope={"id": "s", "label": "QA"},
                        key_terms={"problem": "long contexts in SQuAD"},
                        seed_statement="Long contexts in SQuAD remain unsolved.", evidence=ev,
                        stats={"support": 3})


def _draft(**kw):
    base = dict(reject=False, reject_reason="", title="Long contexts in SQuAD",
                statement="Long contexts in SQuAD remain unsolved in the papers analysed here.",
                why_it_matters="It matters.", why_it_might_not_be_a_gap=["May exist elsewhere."],
                first_steps=["Reproduce E1."], skills=[], data_needs="", compute="unknown", feasibility="medium",
                beginner_friendly=True, evidence_used=["E1"], novelty_queries=["long context QA"])
    base.update(kw)
    return GapDraft.model_validate(base)


def test_foreign_entities():
    assert foreign_entities("Apply GPT-9 to SQuAD", vocab="squad benchmark") == ["GPT-9"]


def test_verify_text_issues():
    c = _cand()
    assert verify_text(_draft(), c) == []
    assert any("unknown evidence ids" in i for i in verify_text(_draft(evidence_used=["E9"]), c))
    assert any("overclaiming" in i for i in verify_text(_draft(statement="No one has ever solved this."), c))
    assert any("why_it_might_not_be_a_gap" in i for i in verify_text(_draft(why_it_might_not_be_a_gap=[]), c))


def _synth_llm(cfg, answers):
    calls = {"n": 0}

    def handler(v):
        i = min(calls["n"], len(answers) - 1)
        calls["n"] += 1
        return answers[i]

    return FakeLLM(cfg, handlers={"synthesize_gap": handler}), calls


def test_synthesize_retry_then_success(tmp_path):
    cfg = make_config(tmp_path)
    bad = _draft(statement="Use GPT-9 for long contexts in SQuAD.").model_dump()
    good = _draft().model_dump()
    llm, calls = _synth_llm(cfg, [bad, good])
    out, stats = synthesize([_cand()], llm, cfg, offline=True)
    assert calls["n"] == 2 and out[0][1].statement == good["statement"] and out[0][2]["synthesized"]


def test_synthesize_two_failures_use_template(tmp_path):
    cfg = make_config(tmp_path)
    bad = _draft(statement="Use GPT-9 for long contexts in SQuAD.").model_dump()
    llm, _ = _synth_llm(cfg, [bad, bad])
    out, stats = synthesize([_cand()], llm, cfg, offline=True)
    assert out[0][2]["synthesized"] is False and stats["text_fallbacks"] == 1
    assert out[0][1].statement == _cand().seed_statement


def test_synthesize_reject_drops(tmp_path):
    cfg = make_config(tmp_path)
    llm, _ = _synth_llm(cfg, [_draft(reject=True, reject_reason="trivial").model_dump()])
    out, stats = synthesize([_cand()], llm, cfg, offline=True)
    assert out == [] and stats["rejected_by_llm"] == 1
