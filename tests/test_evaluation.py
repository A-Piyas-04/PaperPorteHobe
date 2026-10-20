"""Evaluation: sampling, metrics, report, compare harness, telemetry, and the blind Evaluate page."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from conftest import make_config

from scholargrid.evaluation import metrics
from scholargrid.evaluation.rounds import rater_queue
from scholargrid.evaluation.sampling import make_gap_round, stratified


def _cards(n_per=(10, 6, 2)):
    out = []
    for t, n in zip(("stated_open_problem", "evaluation_gap", "missing_bridge"), n_per):
        for i in range(n):
            out.append({"id": f"{t}-{i}", "type": t, "confidence": "medium" if i % 2 else "low",
                        "title": f"{t} {i}", "novelty": {"status": "novel" if i % 3 else "partially_explored"}})
    return out


def test_sampling_strata_overlap_and_seed():
    cards = _cards()
    picked = stratified(cards, 10, seed=1)
    assert len(picked) == 10
    assert sum(1 for c in picked if c["type"] == "missing_bridge") >= 1        # small strata are represented
    r1 = make_gap_round("r", cards, 15, 0.2, seed=3)
    r2 = make_gap_round("r", cards, 15, 0.2, seed=3)
    assert [i["item_id"] for i in r1["items"]] == [i["item_id"] for i in r2["items"]]
    assert sum(i["overlap"] for i in r1["items"]) == 3


def test_rater_queue_overlap_and_rest():
    rnd = {"name": "r", "items": [{"item_id": "a", "overlap": True}, {"item_id": "b", "overlap": False},
                                  {"item_id": "c", "overlap": False}]}
    q1 = [i["item_id"] for i in rater_queue(rnd, "R1", {"b": {"R2"}})]
    assert set(q1) == {"a", "c"}                                   # b was taken by another rater
    assert [i["item_id"] for i in rater_queue(rnd, "R1", {"a": {"R1"}, "b": {"R1"}, "c": {"R1"}})] == []


def _ratings(rows):
    return pd.DataFrame([{"item_id": i, "rater": r, "criteria": c, "comment": ""} for i, r, c in rows])


def test_gap_metrics_majority_novelty_and_kappa():
    items = {"g1": {"type": "a", "confidence": "high", "novelty": {"status": "novel"}},
             "g2": {"type": "a", "confidence": "low", "novelty": {"status": "novel"}},
             "g3": {"type": "b", "confidence": "low", "novelty": {"status": "partially_explored"}}}
    good = {"real": "yes", "open": "yes", "feasible": "yes"}
    solved = {"real": "yes", "open": "no", "feasible": "no"}
    df = _ratings([("g1", "R1", good), ("g1", "R2", good), ("g2", "R1", solved), ("g2", "R2", solved),
                   ("g3", "R1", {"real": "no", "open": "no"}), ("g3", "R2", {"real": "no", "open": "no"})])
    m = metrics.gap_metrics(df, items)
    assert m["gap_precision"] == round(1 / 3, 3)
    assert m["novelty_false_positive_rate"] == 0.5                  # only novel items g1, g2 count
    assert m["precision_by_confidence"]["high"]["precision"] == 1.0
    assert m["kappa_real"] == 1.0
    rng = np.random.default_rng(0)
    rows = [(f"x{i}", r, {"real": str(rng.choice(["yes", "no"])), "open": "yes"}) for i in range(60) for r in ("A", "B")]
    k = metrics.gap_metrics(_ratings(rows), {f"x{i}": {"novelty": {}} for i in range(60)})["kappa_real"]
    assert k is not None and abs(k) < 0.35
    assert metrics.gap_metrics(pd.DataFrame(), {})["gap_precision"] is None


def test_report_writes_launch_bar(tmp_path):
    cfg = make_config(tmp_path)
    rnd = {"name": "pilot", "kind": "gaps", "items": [{"item_id": "g1", "payload": {"type": "a", "confidence": "low",
                                                                                      "novelty": {"status": "novel"}}}]}
    ratings = [{"item_id": "g1", "rater": "R1", "criteria": {"real": "yes", "open": "yes"}, "comment": "nice"}]
    m = metrics.report(cfg, rnd, ratings)
    with open(os.path.join(cfg.abspath(cfg["evaluation"]["report_dir"]), "pilot.md"), encoding="utf-8") as fh:
        md = fh.read()
    assert "Launch bar" in md and ("✅" in md or "❌" in md) and m["gap_precision"] == 1.0
    assert metrics.latest_history(cfg, "gaps")["batch"] == "pilot"


def test_extraction_and_plan_metrics():
    df = _ratings([("problem:1", "R1", {"stated": "yes", "faithful": "yes", "specificity": 3}),
                   ("problem:2", "R1", {"stated": "no", "faithful": "partly", "specificity": 1}),
                   ("facets:p", "R1", {"n_shown": 10, "n_wrong": 1, "n_missing": 1})])
    m = metrics.extraction_metrics(df)
    assert m["problem_precision"] == 0.5 and m["facet_precision"] == 0.9 and m["facet_recall"] == 0.9
    p = metrics.plan_metrics(_ratings([("plan:1", "R1", {"realistic": 4, "grounded": 5, "useful": 3})]))
    assert p["plan_realistic"] == 4.0 and p["n_plans"] == 1


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from scholargrid.pipeline import run

    cfg = make_config(tmp_path_factory.mktemp("evalpipe"), data={"max_papers": 120})
    run(cfg)
    return cfg


def test_compare_harness(built, monkeypatch):
    from scholargrid.evaluation.compare import compare_extract
    from scholargrid.llm import FakeLLM

    cfg = built

    def llm_for(rc):
        route = rc["llm"]["tasks"]["extract"]["route"][0]
        llm = FakeLLM(rc)
        if "lite" in route:                  # the second route finds fewer problems
            from scholargrid.extract.heuristic import heuristic_extract_from_text

            def weaker(v):
                d = heuristic_extract_from_text(str(v["document"])).model_dump()
                d["open_problems"] = d["open_problems"][:1]
                return d

            llm.handlers["extract_paper"] = weaker
        return llm

    out = compare_extract(cfg, ["gemini:gemini-3.8-flash", "gemini:gemini-3.5-flash-lite"], 6, llm_for=llm_for)
    assert [r["route"] for r in out["routes"]] == ["gemini:gemini-3.8-flash", "gemini:gemini-3.5-flash-lite"]
    assert all("requests_per_paper" in r and "truncation_or_fallback_rate" in r for r in out["routes"])
    assert out["routes"][1]["problems_per_paper"] <= out["routes"][0]["problems_per_paper"]
    report = os.path.join(cfg.abspath(cfg["evaluation"]["report_dir"]), f"{out['round']}.compare.md")
    with open(report, encoding="utf-8") as fh:
        assert "Decision rule" in fh.read()


def test_telemetry_track(tmp_path, monkeypatch):
    """track() inserts events synchronously (patched thread), honours opt-out, names and store_queries."""
    import sys

    from conftest import ROOT

    app_dir = os.path.join(ROOT, "app")
    sys.path.insert(0, app_dir)
    import streamlit as st
    import telemetry

    import ui
    from scholargrid.persistence import get_repo

    cfg = make_config(tmp_path, telemetry={"enabled": True})
    repo = get_repo(cfg)
    monkeypatch.setattr(ui, "get_config", lambda: cfg)
    monkeypatch.setattr(ui, "get_repo", lambda: repo)

    class SyncThread:
        def __init__(self, target, args, daemon=True):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(telemetry.threading, "Thread", SyncThread)
    st.session_state.clear() if hasattr(st.session_state, "clear") else None
    telemetry.track("gap_saved", gap_id="g1")
    telemetry.track("not_an_event", x=1)
    names = [e["name"] for e in repo.events(1)]
    assert names == ["gap_saved"]
    props = telemetry.query_props(cfg, "Secret Topic")
    assert "q" not in props and len(props["q_hash"]) == 16
    st.session_state["telemetry_opt_out"] = True
    telemetry.track("gap_saved", gap_id="g2")
    assert len(repo.events(1)) == 1
    st.session_state["telemetry_opt_out"] = False
