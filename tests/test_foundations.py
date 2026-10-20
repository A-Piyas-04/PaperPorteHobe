"""Foundations: survey detection, backbone lift, OpenAlex work lookup, reading paths, stage end-to-end."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
from conftest import make_config

from scholargrid.foundations.backbone import backbone
from scholargrid.foundations.surveys import is_survey


@pytest.mark.parametrize("title,expected", [
    ("A Survey of Retrieval-Augmented Generation", True), ("SoK: Prompt Injection", True),
    ("Tutorial on Diffusion Models", True), ("Peer Review Generation with LLMs", False),
    ("Automated Code Review", False), ("Efficient Attention for Long Documents", False),
])
def test_survey_regex(title, expected):
    assert is_survey(title) is expected


def test_backbone_lift_and_thresholds():
    # area 0: 10 papers cite WALL + WA1; area 1: 10 papers cite WALL + WB1; WRARE cited twice in area 0
    refs = [["WALL", "WA1"] + (["WRARE"] if i < 2 else []) for i in range(10)] + [["WALL", "WB1"]] * 10
    labels = np.array([0] * 10 + [1] * 10)
    bb = backbone(refs, labels, min_citing=3, min_share=0.03, top=10, eps=0.002)
    a0 = bb[0].set_index("openalex_id")
    assert list(bb[0]["openalex_id"])[0] == "WA1"
    assert abs(a0.loc["WALL", "lift"] - 1.0) < 0.01
    assert "WRARE" not in a0.index                       # below min_citing
    strict = backbone(refs, labels, min_citing=1, min_share=0.5, top=10, eps=0.002)
    assert "WRARE" not in set(strict[0]["openalex_id"])   # below min_share


def test_work_lookup_parses_caches_and_respects_deadline(tmp_path, monkeypatch):
    from scholargrid import sources
    from scholargrid.foundations.lookup import WorkLookup

    calls = []

    class Resp:
        def json(self):
            return {"results": [
                {"id": "https://openalex.org/W1", "title": "Attention Is All You Need", "publication_year": 2017,
                 "cited_by_count": 100000, "type": "article", "doi": None,
                 "abstract_inverted_index": {"We": [0], "propose": [1], "transformers": [2]},
                 "locations": [{"landing_page_url": "https://arxiv.org/abs/1706.03762"}],
                 "authorships": [{"author": {"display_name": "A. Vaswani"}}],
                 "primary_location": {"source": {"display_name": "NeurIPS"}}},
                {"id": "https://openalex.org/W2", "title": "BERT", "publication_year": 2018,
                 "cited_by_count": 90000, "type": "article", "doi": "https://doi.org/10.18653/v1/N19-1423",
                 "abstract_inverted_index": None, "locations": [], "authorships": [], "primary_location": {}},
            ]}

    def fake_get(url, params, timeout, headers=None):
        calls.append(params)
        return Resp()

    monkeypatch.setattr(sources, "http_get", fake_get)
    cfg = make_config(tmp_path)
    look = WorkLookup(cfg)
    rows = look.get(["W1", "W2"]).set_index("openalex_id")
    assert calls[0]["filter"] == "ids.openalex:W1|W2"
    assert rows.loc["W1", "abstract"] == "We propose transformers" and rows.loc["W1", "arxiv_id"] == "1706.03762"
    assert rows.loc["W2", "url"] == "https://doi.org/10.18653/v1/N19-1423"
    assert os.path.exists(look.path)
    WorkLookup(cfg).get(["W1", "W2"])
    assert len(calls) == 1                                  # served from cache
    assert WorkLookup(cfg).get(["W9"], deadline=time.time() - 1).empty and len(calls) == 1


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from scholargrid.pipeline import run

    cfg = make_config(tmp_path_factory.mktemp("found"))
    meta, _ = run(cfg)
    return cfg, meta


def test_reading_paths_from_pipeline(built):
    from scholargrid.artifacts import load_bundle

    cfg, meta = built
    b = load_bundle(cfg)
    assert b.reading_paths and len(b.foundations) > 0
    assert meta["foundations"]["pre_2020_foundations"] > 0
    any_survey = False
    for path in b.reading_paths.values():
        ids = [p["paper_id"] for p in path]
        assert len(ids) == len(set(ids))
        years = [p["year"] for p in path if p["step"] == "foundation"]
        assert years == sorted(years)
        assert "WSYNALL01" not in ids[:3]
        any_survey |= any(p["step"] == "survey" and p["paper_id"][-2:] in ("03", "07") for p in path)
        assert all("abstract" not in p for p in path)
    assert any_survey
    first = next(iter(b.reading_paths.values()))
    assert any(p.get("why_read") for p in first)            # FakeLLM reasons attached


def test_disabled_writes_empty_outputs(tmp_path):
    from scholargrid.foundations import build_foundations

    cfg = make_config(tmp_path, foundations={"enabled": False})
    out = tmp_path / "b"
    out.mkdir()
    assert build_foundations(cfg, str(out)) == {"enabled": False}
    assert (out / "reading_paths.json").read_text(encoding="utf-8").strip() == "{}"
