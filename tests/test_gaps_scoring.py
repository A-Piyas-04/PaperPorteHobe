"""Scoring monotonicity and confidence rules."""
from __future__ import annotations

from gaps_helpers import make_ctx, papers_df

from scholargrid.gaps.model import Evidence, GapCandidate, GapCard, NoveltyCheck
from scholargrid.gaps.scoring import confidence, score


def _pair(support=4, status="novel", scope="collection"):
    ev = [Evidence(eid="E1", paper_id="p0", title="t", year=2025, role="states_problem")]
    cand = GapCandidate(type="stated_open_problem", scope={}, key_terms={}, seed_statement="s", evidence=ev,
                        stats={"support": support, "author_groups": support}, specificity=0.7)
    card = GapCard(id="g", type="stated_open_problem", title="t", statement="s", why_it_might_not_be_a_gap=["x"],
                   evidence=ev, novelty=NoveltyCheck(status=status, scope=scope))  # type: ignore[arg-type]
    return card, cand


def test_more_support_more_evidence(tmp_path):
    ctx = make_ctx(tmp_path, papers_df(3, start="2025-01-01"))
    lo = score(*_pair(support=3), ctx, ctx.cfg)
    hi = score(*_pair(support=9), ctx, ctx.cfg)
    assert hi.evidence > lo.evidence and hi.total > lo.total


def test_novelty_order(tmp_path):
    ctx = make_ctx(tmp_path, papers_df(3, start="2025-01-01"))
    totals = [score(*_pair(status=s), ctx, ctx.cfg).total for s in ("novel", "partially_explored", "already_addressed")]
    assert totals[0] > totals[1] > totals[2]


def test_confidence_high_needs_live_check(tmp_path):
    ctx = make_ctx(tmp_path, papers_df(3, start="2025-01-01"))
    card, cand = _pair(support=9, status="novel", scope="collection")
    card.scores = score(card, cand, ctx, ctx.cfg)
    assert confidence(card) == "medium"
    card.novelty = NoveltyCheck(status="novel", scope="collection+live")
    assert confidence(card) == "high"
