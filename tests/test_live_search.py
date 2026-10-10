"""Live multi-source search with mocked sources (no network)."""
import hashlib
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scholargrid import live_search
from scholargrid.acronyms import expand, mine
from scholargrid.bm25 import BM25Index
from scholargrid.live_search import LiveSearcher, fetch_live
from scholargrid.live_store import LiveStore, merge_records
from scholargrid.phrase import PhraseIndex
from scholargrid.sources import RateLimited, SourcePage, make_record, reset_cooldowns


class HashEmbedder:
    """Bag-of-words hashed into 64 dims: similar wording -> similar vectors."""
    fingerprint = "hash-64"

    def _vec(self, text):
        v = np.zeros(64, dtype=np.float32)
        for w in str(text).lower().replace("-", " ").split():
            v[int(hashlib.md5(w.strip(".,;()").encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_query(self, text):
        return self._vec(text)

    def transform(self, texts):
        return np.vstack([self._vec(t) for t in texts])


COLLECTION = [
    ("2401.00001", "Self-admitted technical debt detection",
     "We detect self-admitted technical debt (SATD) in source code comments."),
    ("2401.00002", "Technical debt in machine learning systems",
     "Technical debt accumulates in machine learning pipelines over time."),
    ("2401.00003", "Diffusion models for image synthesis", "We study diffusion models for images."),
    ("2401.00004", "Graph neural networks for molecules", "Message passing on molecular graphs."),
    ("2401.00005", "Repaying self-admitted technical debt",
     "How developers remove self-admitted technical debt (SATD) from code."),
] + [(f"2402.{i:05d}", f"Unrelated topic number {i} on robotics", "Robots grasp objects with arms.")
     for i in range(20)]


def _bundle():
    df = pd.DataFrame(COLLECTION, columns=["arxiv_id", "title", "abstract"])
    df["authors"] = "A. Author"
    df["primary_category"] = "cs.SE"
    df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="7D")
    df["cited_by_count"] = 1
    emb = HashEmbedder()
    texts = (df["title"] + ". " + df["abstract"]).tolist()
    return SimpleNamespace(embedder=emb, embeddings=emb.transform(texts), df=df,
                           labels=np.zeros(len(df), dtype=int), clusters_meta={},
                           bm25=BM25Index.build(texts), index=None, phrase=PhraseIndex.from_frame(df),
                           acronyms=mine(texts), meta={"generated_at": "test"})


def _rec(source, i, title, abstract="", **kw):
    return make_record(source, f"{source}-{i}", title=title, abstract=abstract, date="2023-05-01", **kw)


class FakeAdapter:
    def __init__(self, name, records, delay=0.0, exc=None):
        self.name, self.records, self.delay, self.exc, self.calls = name, records, delay, exc, []

    def fetch(self, phrases, cursor, limit, timeout, cfg, deadline=None):
        self.calls.append((list(phrases), cursor))
        if self.delay:
            time.sleep(self.delay)
        if self.exc:
            raise self.exc
        start = cursor or 0
        chunk = self.records[start:start + limit]
        end = start + len(chunk)
        return SourcePage(self.name, chunk, len(self.records), end if end < len(self.records) else None)


@pytest.fixture
def live_cfg(cfg):
    cfg.raw["search"].update(live=True, live_sources=["arxiv", "openalex", "semantic_scholar"],
                             s2_budget_s=0.5, live_deadline_s=3, live_per_source=3,
                             live_cache_hours=0, cooldown_s=60, live_embed_budget_s=5)
    reset_cooldowns()
    yield cfg
    reset_cooldowns()


def _adapters(monkeypatch, **adapters):
    monkeypatch.setattr(live_search, "get_adapter", lambda name: adapters[name])
    return adapters


def _td_records(source, n, offset=0):
    return [_rec(source, offset + i, f"Technical debt study {offset + i} by {source}",
                 "We analyse technical debt in industry.", doi=f"10.1/{source}.{offset + i}")
            for i in range(n)]


# ---------------------------------------------------------------------------
def test_merge_records_dedups_by_doi_arxiv_and_title_year():
    a = _rec("arxiv", 1, "Paper One", arxiv_id="2401.00001v2")
    b = _rec("openalex", 1, "Paper One (journal version)", doi="https://doi.org/10.48550/arXiv.2401.00001")
    c = _rec("semantic_scholar", 2, "Paper Two", doi="10.1/ABC")
    d = _rec("openalex", 2, "Paper Two", doi="10.1/abc", cited_by_count=9)
    e = make_record("arxiv", "e", title="A study of very specific things", year=2020)
    f = make_record("openalex", "f", title="A Study of Very Specific Things!", year=2021)
    g = make_record("openalex", "g", title="A study of very specific things", year=2015)
    merged = merge_records([a, b, c, d, e, f, g])
    assert len(merged) == 4
    by_title = {m["title"]: m for m in merged}
    assert sorted(by_title["Paper One"]["sources"]) == ["arxiv", "openalex"]
    assert by_title["Paper Two"]["cited_by_count"] == 9


def test_slow_semantic_scholar_falls_back_to_other_sources(live_cfg, monkeypatch):
    _adapters(monkeypatch, arxiv=FakeAdapter("arxiv", _td_records("arxiv", 3)),
              openalex=FakeAdapter("openalex", _td_records("openalex", 3)),
              semantic_scholar=FakeAdapter("semantic_scholar", _td_records("s2", 3), delay=5))
    t0 = time.monotonic()
    out = fetch_live(expand("technical debt"), live_cfg, cache_dir=str(live_cfg.live_dir))
    assert time.monotonic() - t0 < 2.5
    assert out.status == {"arxiv": "ok", "openalex": "ok", "semantic_scholar": "slow"}
    assert len(out.records) == 6
    assert out.totals == {"arxiv": 3, "openalex": 3}


def test_rate_limited_source_cools_down(live_cfg, monkeypatch):
    s2 = FakeAdapter("semantic_scholar", [], exc=RateLimited("429"))
    _adapters(monkeypatch, arxiv=FakeAdapter("arxiv", _td_records("arxiv", 2)),
              openalex=FakeAdapter("openalex", []), semantic_scholar=s2)
    first = fetch_live(expand("technical debt"), live_cfg, cache_dir=str(live_cfg.live_dir))
    assert first.status["semantic_scholar"] == "rate_limited"
    second = fetch_live(expand("technical debt"), live_cfg, cache_dir=str(live_cfg.live_dir))
    assert second.status["semantic_scholar"] == "cooldown"
    assert len(s2.calls) == 1


def test_query_cache_makes_repeat_searches_instant(live_cfg, monkeypatch, tmp_path):
    live_cfg.raw["search"]["live_cache_hours"] = 24
    arx = FakeAdapter("arxiv", _td_records("arxiv", 3))
    _adapters(monkeypatch, arxiv=arx, openalex=FakeAdapter("openalex", []),
              semantic_scholar=FakeAdapter("semantic_scholar", []))
    fetch_live(expand("technical debt"), live_cfg, cache_dir=str(tmp_path))
    again = fetch_live(expand("technical debt"), live_cfg, cache_dir=str(tmp_path))
    assert again.status["arxiv"] == "cached" and len(arx.calls) == 1
    later = fetch_live(expand("technical debt"), live_cfg, cache_dir=str(tmp_path),
                       now=lambda: time.time() + 25 * 3600)
    assert later.status["arxiv"] == "ok" and len(arx.calls) == 2


def test_live_search_merges_sources_into_exact_and_related(live_cfg, monkeypatch):
    arx = FakeAdapter("arxiv", [_rec("arxiv", 1, "Self-admitted technical debt in issue trackers",
                                     "Self-admitted technical debt appears in issues.", arxiv_id="2305.11111")])
    oa = FakeAdapter("openalex", [
        _rec("openalex", 1, "Self-admitted technical debt in issue trackers",
             "Self-admitted technical debt appears in issues.", doi="10.48550/arxiv.2305.11111"),
        _rec("openalex", 2, "Removing technical debt comments from code",
             "Developers remove debt comments in code.", doi="10.1/oa.2"),
        # Already in the collection: must not be duplicated.
        _rec("openalex", 3, "Self-admitted technical debt detection",
             "We detect self-admitted technical debt (SATD) in source code comments.",
             arxiv_id="2401.00001"),
    ])
    _adapters(monkeypatch, arxiv=arx, openalex=oa,
              semantic_scholar=FakeAdapter("semantic_scholar", []))
    searcher = LiveSearcher(live_cfg, _bundle())
    res = searcher.run("SATD")
    assert "self admitted technical debt" in " ".join(res["phrases"])
    exact = [r for r in res["results"] if r["match"] == "exact"]
    ids = [r["paper_id"] for r in res["results"]]
    assert len(ids) == len(set(ids))
    assert ids.count("2401.00001") == 1
    online = next(r for r in exact if r["paper_id"] == "2305.11111")
    assert sorted(online["sources"]) == ["arxiv", "openalex"] and online["is_new"]
    assert {"2401.00001", "2401.00005"} <= {r["paper_id"] for r in exact}
    assert all("robotics" not in r["title"] for r in res["results"])
    assert res["source_totals"]["openalex"] == 3
    assert res["n_found_online"] == 3

    # The found papers are persisted and searchable without the network.
    fresh = LiveSearcher(live_cfg, _bundle(), store=LiveStore.load(live_cfg.live_dir, "hash-64", 64))
    offline = fresh.run("SATD", live=False)
    assert "2305.11111" in {r["paper_id"] for r in offline["results"]}
    assert not offline["results"][0]["is_new"]
    assert fresh.paper("2305.11111")["title"].startswith("Self-admitted")


def test_all_sources_failing_still_returns_collection_results(live_cfg, monkeypatch):
    boom = RuntimeError("network down")
    _adapters(monkeypatch, arxiv=FakeAdapter("arxiv", [], exc=boom),
              openalex=FakeAdapter("openalex", [], exc=boom),
              semantic_scholar=FakeAdapter("semantic_scholar", [], exc=boom))
    res = LiveSearcher(live_cfg, _bundle()).run("technical debt")
    assert set(res["source_status"].values()) == {"error"}
    assert {"2401.00001", "2401.00002", "2401.00005"} <= {
        r["paper_id"] for r in res["results"] if r["match"] == "exact"}


def test_load_more_fetches_the_next_page(live_cfg, monkeypatch):
    arx = FakeAdapter("arxiv", _td_records("arxiv", 7))
    _adapters(monkeypatch, arxiv=arx, openalex=FakeAdapter("openalex", []),
              semantic_scholar=FakeAdapter("semantic_scholar", []))
    searcher = LiveSearcher(live_cfg, _bundle())
    one = searcher.run("technical debt")
    assert one["has_more"] and one["n_found_online"] == 3
    two = searcher.run("technical debt", pages=2)
    assert two["n_found_online"] == 6
    assert [c for _, c in arx.calls] == [None, None, 3]
    online = {r["paper_id"] for r in two["results"] if "arxiv" in r["sources"]}
    assert len(online) == 6


def test_similar_works_for_collection_and_online_papers(live_cfg, monkeypatch):
    _adapters(monkeypatch, arxiv=FakeAdapter("arxiv", _td_records("arxiv", 3)),
              openalex=FakeAdapter("openalex", []), semantic_scholar=FakeAdapter("semantic_scholar", []))
    searcher = LiveSearcher(live_cfg, _bundle())
    searcher.run("technical debt")
    like = searcher.similar("2401.00002", top_k=5)
    assert like["results"] and all(r["paper_id"] != "2401.00002" for r in like["results"])
    online_id = "doi:10.1/arxiv.0"
    assert searcher.paper(online_id) is not None
    like2 = searcher.similar(online_id, top_k=5)
    assert like2["results"] and all(r["paper_id"] != online_id for r in like2["results"])
