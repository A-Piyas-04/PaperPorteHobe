"""Extraction: schema clipping, document builder, heuristic, quote verification, canonicalization,
incremental runs, packing, and quota stop/resume."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_config

from scholargrid.extract.canonical import canonicalize_values
from scholargrid.extract.document import build_document
from scholargrid.extract.heuristic import heuristic_extract_from_text
from scholargrid.extract.schemas import PaperExtraction, StatedProblem
from scholargrid.extract.store import ExtractionStore
from scholargrid.extract.verify import verify_quotes
from scholargrid.fulltext.synthetic import synthetic_sections
from scholargrid.llm import FakeLLM, LLMInvalidOutput, QuotaExhausted

BUDGET = {"abstract": 3000, "introduction": 3500, "experiments": 3500, "limitations": 4000,
          "future_work": 3000, "conclusion": 3000}


def _ext(**kw):
    base = dict(one_line_summary="x", tasks=[], methods=[], datasets=[], metrics=[], domains=[], languages=[],
                modalities=[], contribution_types=[],
                evaluation={k: None for k in ("real_world", "human_eval", "robustness", "efficiency_reported",
                                              "low_resource", "multilingual")}, open_problems=[], findings=[],
                code_url=None, data_released=None, compute="unknown")
    base.update(kw)
    return PaperExtraction.model_validate(base)


def _secs_frame(sections):
    return pd.DataFrame([s.to_row("p1", "synthetic") for s in sections])


def _synth_doc(title="Sequence: attention for language models", category="cs.CL"):
    abstract = "We study sequence modelling in the context of language models. Experiments show gains."
    secs = synthetic_sections("synth.00001", title, abstract, category)
    return build_document("synth.00001", title, abstract, _secs_frame(secs), pd.DataFrame(), BUDGET, 24000, 25)


# 1. schema clipping -------------------------------------------------------------------------------------
def test_schema_clipping():
    p = StatedProblem(statement="w " * 600, quote="q", section="[S5]", problem_type="future_work", specificity=7)
    assert len(p.statement) <= 320 and p.specificity == 3 and p.section == "S5"
    assert _ext(code_url="github.com/x").code_url is None
    assert _ext(code_url="https://github.com/x").code_url == "https://github.com/x"
    assert _ext(tasks=["A", "a", " ", "b."]).tasks == ["A", "b"]


# 2. document builder ------------------------------------------------------------------------------------
def test_document_builder_markers_budget_and_abstract_only():
    doc = _synth_doc()
    assert "[S3] LIMITATIONS: Limitations" in doc.text and doc.has_fulltext
    assert set(doc.sources) >= {"S0", "S3", "S4"} and doc.kinds["S3"] == "limitations"
    tiny = build_document("p", "T", "abstract text", _secs_frame(synthetic_sections("p", "A: b", "x " * 3000, "cs.CL")),
                          pd.DataFrame(), {"abstract": 50}, 2000, 0)
    assert len(tiny.text) <= 2000
    only = build_document("p", "Title", "Just an abstract.", pd.DataFrame(), pd.DataFrame(), BUDGET, 24000, 25)
    assert "[S0] ABSTRACT" in only.text and not only.has_fulltext


# 3. heuristic on synthetic full text --------------------------------------------------------------------
def test_heuristic_on_synthetic_fulltext():
    doc = _synth_doc()
    ext = heuristic_extract_from_text(doc.text)
    assert ext.open_problems
    verified, dropped = verify_quotes(ext, doc, 90, 6)
    assert verified and dropped == 0 and all(v.ratio == 100.0 for v in verified)
    assert any(d.startswith("Synth") for d in ext.datasets)
    assert ext.methods[0].role == "proposed"


def test_fake_handler_strips_tags():
    from scholargrid.llm.prompts import wrap_untrusted

    doc = _synth_doc()
    a = heuristic_extract_from_text(doc.text)
    b = heuristic_extract_from_text(wrap_untrusted("paper", doc.text))
    assert a.model_dump() == b.model_dump()


# 4. quote verification ----------------------------------------------------------------------------------
def test_quote_verification_exact_fuzzy_and_fabricated():
    source = 'Our parser does not handle "code-switched" input. It is limited to English newswire.'
    from scholargrid.extract.document import PaperDocument

    doc = PaperDocument("p", source, {"S1": source}, {"S1": "S1"}, {"S1": "limitations"}, True)
    probs = [
        {"statement": "s", "quote": "It is limited to English newswire.", "section": "S1",
         "problem_type": "limitation_of_this_work", "specificity": 2},
        {"statement": "s", "quote": "Our parser does not  handle \u201ccode-switched\u201d input.", "section": "S9",
         "problem_type": "limitation_of_this_work", "specificity": 3},
        {"statement": "s", "quote": "The model was never evaluated on medical images at all.", "section": "S1",
         "problem_type": "future_work", "specificity": 2},
    ]
    kept, dropped = verify_quotes(_ext(open_problems=probs), doc, 90, 6)
    assert dropped == 1 and len(kept) == 2
    fuzzy = next(k for k in kept if "code-switched" in k.quote)
    assert fuzzy.quote in source and fuzzy.section_label == "S1"
    exact = next(k for k in kept if "newswire" in k.quote)
    assert exact.ratio == 100.0


# 5. canonicalization ------------------------------------------------------------------------------------
def test_canonicalization_guards():
    counts = {"LLMs": 5, "large language models": 9, "Large Language Model": 2, "BERT": 4, "RoBERTa": 3}
    acr = {"llm": {"large language models": 7}}
    vecs = np.eye(5, dtype=np.float32)
    vecs[4] = 0.95 * vecs[3] + np.sqrt(1 - 0.95 ** 2) * np.eye(5, dtype=np.float32)[0]  # cos(BERT, RoBERTa) = 0.95
    m = canonicalize_values(counts, vecs, acr, 0.88, 80)
    assert m["LLMs"] == m["Large Language Model"] == "large language models"
    assert m["BERT"] != m["RoBERTa"]


# 6-8. run_extract ---------------------------------------------------------------------------------------
@pytest.fixture
def small(tmp_path):
    from scholargrid.data_ingest import build_corpus
    from scholargrid.fulltext import run_fulltext

    cfg = make_config(tmp_path, data={"max_papers": 24})
    df = build_corpus(cfg)
    run_fulltext(df, cfg)
    return cfg, df


def test_run_extract_fake_incremental_and_prompt_version(small, monkeypatch):
    from scholargrid.extract import run_extract
    from scholargrid.llm.prompts import PROMPTS

    cfg, df = small
    s1 = run_extract(df, cfg)
    assert s1["llm"] == len(df) == s1["in_scope"] and s1["problems_verified"] > 0
    assert set(ExtractionStore(cfg).df["model"]) == {"fake:fake"}
    assert run_extract(df, cfg)["done_now"] == 0
    from dataclasses import replace

    monkeypatch.setitem(PROMPTS, "extract_paper", replace(PROMPTS["extract_paper"], version="2"))
    assert run_extract(df, cfg)["done_now"] == len(df)


def test_heuristic_mode_without_llm(small):
    from scholargrid.extract import run_extract

    cfg, df = small
    cfg.raw["llm"]["provider"] = "none"
    s = run_extract(df, cfg)
    assert s["heuristic"] == len(df) and s["llm"] == 0
    assert set(ExtractionStore(cfg).df["mode"]) == {"heuristic"}


class QuotaAfter(FakeLLM):
    """FakeLLM whose structured_many answers the first ``n_ok`` items, then reports quota exhausted."""

    def __init__(self, cfg, n_ok):
        super().__init__(cfg)
        self.n_ok = n_ok
        self.answered = 0

    def structured_many(self, task, prompt, schema, items, **kw):
        out = {}
        for iid, v in items:
            if self.answered < self.n_ok:
                out[iid] = self.structured(task, prompt, schema, **v)
                self.answered += 1
            else:
                out[iid] = QuotaExhausted("daily limit")
        return out


def test_quota_stop_heuristic_then_upgrade(small):
    from scholargrid.extract import run_extract

    cfg, df = small
    n = len(df)
    s = run_extract(df, cfg, llm=QuotaAfter(cfg, 2))
    assert s["quota_stopped"] and s["llm"] == 2 and s["heuristic"] == n - 2 and s["pending_llm"] == n - 2
    s2 = run_extract(df, cfg, llm=FakeLLM(cfg))
    assert s2["done_now"] == n - 2 and s2["llm"] == n - 2
    assert set(ExtractionStore(cfg).df["mode"]) == {"llm"}


def test_quota_stop_skip_and_stop(small):
    from scholargrid.extract import run_extract

    cfg, df = small
    cfg.raw["extract"]["on_quota"] = "skip"
    s = run_extract(df, cfg, llm=QuotaAfter(cfg, 2))
    assert s["llm"] == 2 and len(ExtractionStore(cfg).df) == 2
    ExtractionStore(cfg).drop(df["arxiv_id"].astype(str))
    cfg.raw["extract"]["on_quota"] = "stop"
    with pytest.raises(QuotaExhausted):
        run_extract(df, cfg, llm=QuotaAfter(cfg, 2))
    assert len(ExtractionStore(cfg).df) == 2


class AlwaysInvalid(FakeLLM):
    def structured_many(self, task, prompt, schema, items, **kw):
        self.calls.append(len(items))
        return {iid: LLMInvalidOutput("bad json") for iid, _ in items}


def test_failing_paper_stops_retrying(small):
    from scholargrid.extract import run_extract

    cfg, df = small
    cfg.raw["extract"]["max_llm_attempts"] = 2
    client = AlwaysInvalid(cfg)
    for _ in range(2):
        run_extract(df, cfg, llm=client)
    assert set(ExtractionStore(cfg).df["llm_attempts"]) == {2}
    assert run_extract(df, cfg, llm=client)["done_now"] == 0 and len(client.calls) == 2


def test_packing_with_router_retries_missing_item(small, tmp_path):
    from scholargrid.extract import run_extract
    from scholargrid.extract.prompts import _fake_extract
    from scholargrid.llm.backends import RawResult
    from scholargrid.llm.cache import DiskCache
    from scholargrid.llm.quota import QuotaManager
    from scholargrid.llm.router import LLMRouter
    from scholargrid.llm.usage import UsageLedger

    cfg, df = small
    df = df.head(3)
    cfg.raw["llm"]["tasks"]["extract"]["pack_size"] = 3

    class Backend:
        calls = 0

        def generate(self, model, system, user, schema, **kw):
            Backend.calls += 1
            import re

            docs = re.findall(r'<item id="([^"]+)">\n(.*?)\n</item>', user, re.S)
            if docs:   # packed: answer all but the last item
                results = [{"item_id": iid, **_fake_extract({"document": body})} for iid, body in docs[:-1]]
                return RawResult(json.dumps({"results": results}), {"input_tokens": 1})
            return RawResult(json.dumps(_fake_extract({"document": user})), {"input_tokens": 1})

    router = LLMRouter(cfg, {"gemini": Backend()}, QuotaManager(str(tmp_path / "q"), {}, {}),
                       UsageLedger(str(tmp_path / "q"), {}, None), DiskCache(str(tmp_path / "c")))
    s = run_extract(df, cfg, llm=router)
    assert s["llm"] == 3 and Backend.calls == 2 and s["requests_used"] == 2


def test_extraction_bundle_tables(small, tmp_path):
    from scholargrid.embeddings import Embedder, paper_text
    from scholargrid.extract import run_extract
    from scholargrid.extract.bundle import write_extraction_bundle

    cfg, df = small
    run_extract(df, cfg)
    emb = Embedder("tfidf", cfg)
    emb.fit_transform(paper_text(df))
    out = tmp_path / "bundle"
    out.mkdir()
    stats = write_extraction_bundle(cfg, df, emb, str(out))
    problems = pd.read_parquet(out / "problems.parquet")
    assert stats["problems"] == len(problems) > 0
    assert len(np.load(out / "problem_embeddings.npy")) == len(problems)
    vocab = json.loads((out / "facet_vocab.json").read_text(encoding="utf-8"))
    assert vocab["dataset"]
