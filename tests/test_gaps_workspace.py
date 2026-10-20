"""Online gap cards for a topic workspace: generated once, then served from the cache."""
from __future__ import annotations

import pytest
from conftest import make_config


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from scholargrid.artifacts import load_bundle
    from scholargrid.pipeline import run

    cfg = make_config(tmp_path_factory.mktemp("gws"))
    run(cfg)
    return cfg, load_bundle(cfg)


def test_gaps_for_workspace_cached(built):
    from scholargrid.gaps import gaps_for_workspace
    from scholargrid.llm import FakeLLM
    from scholargrid.workspace import build_workspace

    cfg, bundle = built
    ws = build_workspace("language models", bundle, None, cfg, bundle_version="v", llm=FakeLLM(cfg))
    llm = FakeLLM(cfg)
    data = gaps_for_workspace(ws, bundle, None, cfg, llm)
    assert data["cards"] and all(c["why_it_might_not_be_a_gap"] for c in data["cards"])
    assert all(c["novelty"]["status"] != "not_checked" for c in data["cards"])
    assert len(data["cards"]) <= cfg["gaps"]["online_max_new_cards"]
    n_calls = len(llm.calls)
    again = gaps_for_workspace(ws, bundle, None, cfg, llm)
    assert len(llm.calls) == n_calls and again["cards"] == data["cards"]


def test_offline_gaps_in_bundle(built):
    cfg, bundle = built
    cards = [c for cs in bundle.gaps["areas"].values() for c in cs]
    assert cards
    quotes = set(bundle.problems["quote"])
    for c in cards:
        assert c["evidence"] and c["why_it_might_not_be_a_gap"]
        assert c["novelty"]["status"] != "not_checked"
        assert all(e["quote"] in quotes for e in c["evidence"] if e["quote"])
