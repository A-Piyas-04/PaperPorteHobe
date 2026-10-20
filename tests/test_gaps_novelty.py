"""Novelty check: planted 'already addressed' paper, heuristic mode, live path; finalize drops/keeps."""
from __future__ import annotations

import numpy as np
from gaps_helpers import make_ctx, papers_df, unit

from scholargrid.gaps.engine import finalize
from scholargrid.gaps.model import Evidence, GapCandidate, GapCard
from scholargrid.gaps.novelty import novelty_check
from scholargrid.llm import FakeLLM, NullLLM


class StubEmbedder:
    def transform(self, texts):
        return np.tile(unit([1, 0, 0, 0]), (len(texts), 1))


def _ctx(tmp_path, **kw):
    papers = papers_df(4, start="2025-01-01", months_apart=2)
    papers.loc[3, "abstract"] = "We ADDRESSES_GAP by handling long documents directly."
    vecs = np.vstack([unit([0, 1, 0, 0]), unit([0, 0, 1, 0]), unit([0, 0, 0, 1]), unit([1, 0.05, 0, 0])])
    return make_ctx(tmp_path, papers, vectors=vecs, **kw)


def _card():
    ev = [Evidence(eid="E1", paper_id="p0", title="Paper p0", role="states_problem")]
    return GapCard(id="g1", type="stated_open_problem", title="t", statement="Long documents are not handled.",
                   why_it_might_not_be_a_gap=["x"], evidence=ev, key_terms={"problem": "long documents"})


def test_planted_hit_is_already_addressed(tmp_path):
    ctx = _ctx(tmp_path)
    nc = novelty_check(_card(), ctx, StubEmbedder(), FakeLLM(ctx.cfg), ctx.cfg)
    assert nc.status == "already_addressed" and nc.method == "llm" and nc.scope == "collection"
    assert any(h.paper_id == "p3" and h.verdict == "addresses" for h in nc.hits)


def test_heuristic_never_addresses(tmp_path):
    ctx = _ctx(tmp_path)
    nc = novelty_check(_card(), ctx, StubEmbedder(), NullLLM(ctx.cfg), ctx.cfg)
    assert nc.method == "heuristic" and nc.status != "already_addressed"


def test_live_path_includes_live_hits(tmp_path):
    ctx = _ctx(tmp_path)

    class Searcher:
        live_enabled = True
        store = None

        def run(self, q, pages=1, live=True):
            return {"results": [{"paper_id": "live1", "title": "Long documents handled", "date": "2025-06-01",
                                 "abstract": "Long documents are handled.", "url": "https://x", "bundle_index": None}]}

    nc = novelty_check(_card(), ctx, StubEmbedder(), FakeLLM(ctx.cfg), ctx.cfg, searcher=Searcher(), live=True)
    assert nc.scope == "collection+live" and any(h.paper_id == "live1" for h in nc.hits)


def _cand():
    ev = [Evidence(eid="E1", paper_id="p0", title="Paper p0", role="states_problem")]
    return GapCandidate(type="stated_open_problem", scope={"kind": "test", "id": "t", "label": "T"},
                        key_terms={"problem": "long documents"}, seed_statement="Long documents are not handled.",
                        evidence=ev, stats={"support": 4, "author_groups": 4}, specificity=0.7)


def test_finalize_drops_or_keeps_already_addressed(tmp_path):
    ctx = _ctx(tmp_path)
    cards, already, stats = finalize([(_cand(), ctx)], StubEmbedder(), FakeLLM(ctx.cfg), ctx.cfg, offline=True)
    assert cards == [] and already == [] and stats["already_addressed"] == 1
    ctx2 = _ctx(tmp_path, keep_already_addressed=True)
    cards, already, _ = finalize([(_cand(), ctx2)], StubEmbedder(), FakeLLM(ctx2.cfg), ctx2.cfg, offline=True)
    assert cards == [] and len(already) == 1
