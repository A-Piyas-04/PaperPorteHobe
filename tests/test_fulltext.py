"""Full text: heading classifier, cue finder, HTML/PDF parsing, fetch fallback, store and resume."""
from __future__ import annotations

import os

import pytest
from conftest import ROOT, make_config

from scholargrid.fulltext.model import Section
from scholargrid.fulltext.parse_html import parse_html
from scholargrid.fulltext.parse_pdf import split_pdf_text
from scholargrid.fulltext.sections import classify_heading, find_cues, split_sentences

KEEP = {"abstract", "introduction", "discussion", "limitations", "future_work", "conclusion"}


@pytest.mark.parametrize("heading,kind", [
    ("5 Limitations", "limitations"), ("Conclusion and Future Work", "future_work"),
    ("Limitations and Future Work", "limitations"), ("7. Concluding Remarks", "conclusion"),
    ("Threats to Validity", "threats_to_validity"), ("A Appendix: Proofs", "appendix"),
    ("4.2 Experimental Setup", "experiments"), ("Related Work", "related_work"),
    ("Broader Impact Statement", "broader_impact"), ("Strange Title", "other"),
])
def test_classify(heading, kind):
    assert classify_heading(heading) == kind


def test_parse_html_nested_limitations():
    with open(os.path.join(ROOT, "tests", "data", "latexml_sample.html"), encoding="utf-8") as fh:
        html = fh.read()
    secs = parse_html(html, KEEP, 8000)
    kinds = [s.kind for s in secs]
    assert kinds[0] == "abstract" and "limitations" in kinds and "future_work" in kinds
    lim = next(s for s in secs if s.kind == "limitations")
    assert lim.parent_idx is not None and lim.anchor == "S5.SS1"
    assert all("Bibliography" not in s.heading for s in secs)
    text = " ".join(s.text for s in secs)
    assert "x^2" in text and "[CIT]" in text and "footnote" not in text
    disc = next(s for s in secs if s.kind == "discussion")
    assert "code-switched" not in disc.text          # nested subsection text stays in its own record


def test_split_sentences_abbrev():
    sents = [s for _, s in split_sentences("We use e.g. BERT. It works. Future work is open.")]
    assert sents == ["We use e.g. BERT.", "It works.", "Future work is open."]


def test_cues():
    s = [Section(0, None, "Conclusion", "conclusion",
                 "Our method does not handle long documents. It is fast. We leave multilingual data for future work.")]
    names = [c.cue for c in find_cues(s)]
    # first matching pattern wins: "future work" is caught by `future_work` before `left_for_future`
    assert names == ["cannot", "future_work"]


def test_every_limitations_sentence_is_a_candidate():
    s = [Section(0, None, "Limitations", "limitations", "We only tested on two small datasets from one year.")]
    assert [c.cue for c in find_cues(s)] == ["in_limitations"]


def test_pdf_split():
    with open(os.path.join(ROOT, "tests", "data", "pdf_text_sample.txt"), encoding="utf-8") as fh:
        text = fh.read()
    blocks = split_pdf_text(text)
    heads = [h for h, _ in blocks]
    assert "Limitations" in heads and "References" in heads and "Appendix" in heads
    body = dict(blocks)["Limitations"]
    assert "code-switched" in body


def test_fetch_fallback_order(tmp_path, monkeypatch):
    """arxiv_html 404 -> ar5iv returns non-LaTeXML (redirect) -> pdf ok."""
    from scholargrid.fulltext import fetch as F

    cfg = make_config(tmp_path)
    calls = []

    def fake_get(self, url):
        calls.append(url)
        if "arxiv.org/html" in url:
            return 404, b"", ""
        if "ar5iv" in url:
            return 200, b"<html>abs page</html>", ""
        return 200, b"%PDF", ""

    monkeypatch.setattr(F.Fetcher, "_get", fake_get)
    monkeypatch.setattr(F, "parse_pdf", lambda *a, **k: [Section(0, None, "Limitations", "limitations", "x" * 600)])
    res = F.Fetcher(cfg).fetch("2401.00001")
    assert res.status == "ok" and res.source == "pdf" and len(calls) == 3


def test_fetch_all_fail_is_error(tmp_path, monkeypatch):
    from scholargrid.fulltext import fetch as F

    cfg = make_config(tmp_path)
    monkeypatch.setattr(F.Fetcher, "_get", lambda self, url: (None, b"", "ConnectionError: down"))
    res = F.Fetcher(cfg).fetch("2401.00002")
    assert res.status == "error" and "ConnectionError" in res.error


def test_run_fulltext_synthetic_and_resume(tmp_path):
    from scholargrid.data_ingest import build_corpus
    from scholargrid.fulltext import FulltextStore, run_fulltext

    cfg = make_config(tmp_path)
    df = build_corpus(cfg)
    s1 = run_fulltext(df, cfg)
    assert s1["ok"] == len(df) and s1["fetched_now"] == len(df) and s1["coverage"] == 1.0
    assert run_fulltext(df, cfg)["fetched_now"] == 0          # resumable: nothing new to fetch
    store = FulltextStore(cfg)
    secs = store.sections()
    assert {"limitations", "future_work"} <= set(secs["kind"])
    assert len(store.cues()) > 0
    store.compact()
    assert len(FulltextStore(cfg).sections()) == len(secs)


def test_disabled_writes_nothing(tmp_path):
    from scholargrid.data_ingest import build_corpus
    from scholargrid.fulltext import run_fulltext

    cfg = make_config(tmp_path, fulltext={"enabled": False})
    stats = run_fulltext(build_corpus(cfg), cfg)
    assert stats["enabled"] is False and stats["ok"] == 0


def test_schema_enforces_polite_interval(tmp_path):
    from scholargrid.schema import ConfigError, validate_config

    raw = make_config(tmp_path).raw
    raw["fulltext"]["request_interval_s"] = 1.0
    with pytest.raises(ConfigError):
        validate_config(raw)
