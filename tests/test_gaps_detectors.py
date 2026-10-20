"""Gap detectors on hand-built tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from gaps_helpers import facets_rows, make_ctx, papers_df, problems_df, unit

from scholargrid.gaps.detectors import (
    detect_bridge,
    detect_combinatorial,
    detect_evaluation,
    detect_stated,
    detect_unsettled,
)


def _stated_ctx(tmp_path, first_date="2022-01-01", extra_papers=0, later_vec=None, months_apart=2, **kw):
    papers = papers_df(6 + extra_papers, start=first_date, months_apart=months_apart)
    base = unit([1, 0, 0, 0, 0, 0, 0, 0])
    near = [unit(base + 0.1 * unit([0, 1, 0, 0, 0, 0, 0, 0]) * s) for s in (0.1, 0.2, 0.15, 0.05)]
    far = [unit([0, 0, 1, 0, 0, 0, 0, 0]), unit([0, 0, 0, 1, 0, 0, 0, 0])]
    pvec = np.vstack(near + far)
    probs = problems_df([{"arxiv_id": f"p{i}", "statement": f"Long documents are not handled ({i})",
                          "quote": f"Our method does not handle long documents ({i})."} for i in range(4)]
                        + [{"arxiv_id": "p4", "statement": "Unrelated A", "quote": "Something else entirely here."},
                           {"arxiv_id": "p5", "statement": "Unrelated B", "quote": "Another unrelated limitation."}])
    vectors = np.vstack([unit(np.eye(8)[i % 8]) for i in range(len(papers))]).astype(np.float32)
    if later_vec is not None:
        vectors[-1] = later_vec
    return make_ctx(tmp_path, papers, problems=probs, problem_vecs=pvec, vectors=vectors, **kw), probs


def test_stated_cluster_support_and_quotes(tmp_path):
    ctx, probs = _stated_ctx(tmp_path, first_date="2023-06-01")
    cands = detect_stated(ctx)
    assert len(cands) == 1
    c = cands[0]
    assert c.stats["support"] == 4 and c.type == "stated_open_problem"
    quotes = {e.quote for e in c.evidence if e.role == "states_problem"}
    assert quotes <= set(probs["quote"]) and len(quotes) == 4
    ctx2, _ = _stated_ctx(tmp_path, first_date="2023-06-01", min_support=5)
    assert detect_stated(ctx2) == []


def test_stated_long_standing(tmp_path):
    # first statement 2022-01, last 2023-07; ref date set so one statement is within 12 months and first >= 24 months
    ctx, _ = _stated_ctx(tmp_path, first_date="2022-01-01", months_apart=6, ref_date="2024-03-01")
    assert detect_stated(ctx)[0].type == "long_standing_problem"


def test_stated_related_attempt(tmp_path):
    later = unit([1, 0.05, 0, 0, 0, 0, 0, 0])
    ctx, _ = _stated_ctx(tmp_path, first_date="2023-01-01", extra_papers=1, later_vec=later)
    c = detect_stated(ctx)[0]
    assert any(e.role == "related_attempt" and e.paper_id == "p6" for e in c.evidence)


def _combo_ctx(tmp_path, sims_high=True, **kw):
    papers = papers_df(40)
    rows = []
    for i in range(12):
        rows += [(f"p{i}", "method", "M", "proposed"), (f"p{i}", "task", "T1" if i < 8 else "T1b", "")]
    for i in range(12, 24):
        rows += [(f"p{i}", "method", f"Other{i % 3}", "proposed"), (f"p{i}", "task", "T2", "")]
    for i in range(24, 40):
        rows += [(f"p{i}", "method", f"Other{i % 3}", "proposed"), (f"p{i}", "task", "T3", "")]
    e = np.eye(6, dtype=np.float32)
    if sims_high:
        t1 = unit(e[0])
        vv = {"task": {"T1": t1, "T1b": unit(0.8 * e[0] + 0.6 * e[1]), "T2": unit(0.7 * e[0] + 0.714 * e[2]),
                       "T3": unit(e[3])}}
    else:
        vv = {"task": {"T1": unit(e[0]), "T1b": unit(0.1 * e[0] + e[1]), "T2": unit(0.1 * e[0] + e[2]),
                       "T3": unit(e[3])}}
    vv["method"] = {"M": unit(e[4]), "Other0": unit(e[5]), "Other1": unit(e[5] + 0.1 * e[4]),
                    "Other2": unit(e[5] - 0.1 * e[4])}
    return make_ctx(tmp_path, papers, facets=facets_rows(rows), value_vectors=vv, combo_pairs=["method:task"], **kw)


def test_combinatorial_with_bridge(tmp_path):
    cands = detect_combinatorial(_combo_ctx(tmp_path))
    hit = [c for c in cands if c.key_terms == {"method": "M", "task": "T2"}]
    assert hit, [c.key_terms for c in cands]
    c = hit[0]
    assert c.stats["observed"] == 0 and c.stats["p_value"] <= 0.05 and c.stats["bridge"] == "T1"
    assert any(e.role == "bridge" for e in c.evidence)


def test_combinatorial_needs_bridge(tmp_path):
    cands = detect_combinatorial(_combo_ctx(tmp_path, sims_high=False))
    assert not [c for c in cands if c.key_terms == {"method": "M", "task": "T2"}]


def _eval_ctx(tmp_path, real_world):
    papers = papers_df(8)
    rows = [(f"p{i}", "method", "M", "proposed") for i in range(6)] + [(f"p{i}", "dataset", "D1", "") for i in range(6)]
    profiles = pd.DataFrame({"arxiv_id": [f"p{i}" for i in range(6)], "eval_real_world": real_world,
                             "eval_robustness": [None] * 6, "compute": ["unknown"] * 6})
    return make_ctx(tmp_path, papers, facets=facets_rows(rows), profiles=profiles)


def test_evaluation_triggers(tmp_path):
    trig = {c.stats["trigger"] for c in detect_evaluation(_eval_ctx(tmp_path, [False] * 6))}
    assert {"narrow_benchmarks", "no_real_world"} <= trig and "no_robustness" not in trig
    trig2 = {c.stats["trigger"] for c in detect_evaluation(_eval_ctx(tmp_path, [False] + [None] * 5))}
    assert "no_real_world" not in trig2 and "narrow_benchmarks" in trig2


def _findings(directions):
    return pd.DataFrame({"finding_id": ["f0", "f1"], "arxiv_id": ["p0", "p1"],
                         "claim": ["X improves accuracy", "X does not help"], "task": ["question answering"] * 2,
                         "dataset": ["SQuAD"] * 2, "metric": ["F1"] * 2, "direction": directions,
                         "compared_to": ["BERT baseline", "the BERT baseline"]})


def test_unsettled(tmp_path):
    papers = papers_df(3)
    cands = detect_unsettled(make_ctx(tmp_path, papers, findings=_findings(["improves", "no_effect"])))
    assert len(cands) == 1 and all(e.note.startswith("Reported (paraphrased)") for e in cands[0].evidence)
    assert detect_unsettled(make_ctx(tmp_path, papers, findings=_findings(["improves", "improves"]))) == []


def _bridge_ctx(tmp_path, cross: bool):
    n = 20
    papers = papers_df(n)
    papers["openalex_id"] = [f"W{i}" for i in range(n)]
    labels = np.array([0] * 10 + [1] * 10)
    refs = []
    for i in range(n):
        if i % 2:                                   # 50% coverage
            refs.append("")
            continue
        own = [f"W{j}" for j in (range(10) if i < 10 else range(10, 20)) if j != i][:4]
        other = [f"W{j}" for j in (range(10, 20) if i < 10 else range(10))][:4] if cross else []
        refs.append(" ".join(own + other))
    papers["references"] = refs
    vecs = np.vstack([unit([1, 0.05 * (i % 3), 0]) if i < 10 else unit([1, -0.05 * (i % 3), 0.1])
                      for i in range(n)])
    return make_ctx(tmp_path, papers, vectors=vecs, sub_labels=labels, sub_names={0: "Alpha", 1: "Beta"},
                    bridge_sim_quantile=0.5)


def test_bridge(tmp_path):
    cands = detect_bridge(_bridge_ctx(tmp_path, cross=False))
    assert len(cands) == 1 and cands[0].key_terms == {"area_a": "Alpha", "area_b": "Beta"}
    assert {e.role for e in cands[0].evidence} == {"side_a", "side_b"}
    assert detect_bridge(_bridge_ctx(tmp_path, cross=True)) == []
