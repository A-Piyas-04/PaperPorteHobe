import json
from datetime import date

import pandas as pd
import pytest

from conftest import make_config
from scholargrid.data_ingest import (IngestError, build_corpus, clean_records, iter_kaggle,
                                     parse_atom_entry, sample_corpus)
from scholargrid.oai_pmh import harvest, harvest_range, parse_page

ABSTRACT = ("We propose a method for learning representations of graphs and show that it improves "
            "accuracy on several benchmarks while being efficient. ") * 2


def _row(aid, title="A study of graphs", cats="cs.LG", when="2024-03-01", **kw):
    return {"arxiv_id": aid, "title": title, "abstract": ABSTRACT, "authors": "A. B",
            "categories": cats, "primary_category": cats.split()[0], "first_submitted": when, **kw}


def test_clean_dedups_versions_titles_and_filters(cfg):
    df = pd.DataFrame([
        _row("2403.00001v1", title="Graph learning", last_updated="2024-03-01"),
        _row("2403.00001v2", title="Graph learning (revised)", last_updated="2024-04-01"),
        _row("2403.00002v1", title="GRAPH learning (revised)!"),       # duplicate title
        _row("2403.00003v1", title="Physics", cats="physics.comp-ph"),  # not cs
        _row("2403.00004v1", title="Old", when="2019-01-01"),           # outside window
        {**_row("2403.00005v1", title="Short"), "abstract": "too short"},
    ])
    out = clean_records(df, cfg)
    assert list(out["arxiv_id"]) == ["2403.00001"]
    assert out.iloc[0]["title"] == "Graph learning (revised)"
    assert {"first_submitted", "last_updated", "year_month", "sample_weight"} <= set(out.columns)


def test_stratified_sample_keeps_small_strata_and_weights(cfg):
    rows = []
    for i in range(900):
        rows.append(_row(f"2403.{i:05d}", title=f"big {i}", cats="cs.LG"))
    for i in range(30):
        rows.append(_row(f"2404.{i:05d}", title=f"small {i}", cats="cs.DL", when="2024-04-02"))
    clean = clean_records(pd.DataFrame(rows), cfg)
    cfg.raw["data"]["max_papers"] = 200
    s = sample_corpus(clean, cfg)
    assert len(s) == 200
    assert (s["primary_category"] == "cs.DL").sum() == 30          # small stratum kept whole
    big = s[s["primary_category"] == "cs.LG"]
    assert big["sample_weight"].iloc[0] == pytest.approx(900 / len(big))
    assert s["sample_weight"].mul(1).sum() == pytest.approx(930)


def test_failed_source_raises_without_fallback(tmp_path):
    cfg = make_config(tmp_path)
    cfg.raw["data"].update(source="kaggle", allow_synthetic_fallback=False,
                           kaggle_json=str(tmp_path / "missing.json"))
    with pytest.raises(IngestError, match="Kaggle snapshot not found"):
        build_corpus(cfg)


def test_failed_source_falls_back_only_when_allowed(tmp_path):
    cfg = make_config(tmp_path)
    cfg.raw["data"].update(source="kaggle", allow_synthetic_fallback=True,
                           kaggle_json=str(tmp_path / "missing.json"))
    df = build_corpus(cfg)
    assert len(df) > 0 and df.attrs["fallback_from"] == "kaggle"


def test_kaggle_streams_whole_file(tmp_path):
    cfg = make_config(tmp_path)
    cfg.raw["data"].update(date_start="2024-01-01", date_end="2024-12-31")
    path = tmp_path / "snap.json"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(500):  # old papers first, like the real snapshot
            fh.write(json.dumps({"id": f"0704.{i:04d}", "categories": "cs.LG", "title": f"old {i}",
                                 "abstract": ABSTRACT, "authors_parsed": [["Doe", "J", ""]],
                                 "versions": [{"created": "Mon, 2 Apr 2007 19:18:42 GMT"}]}) + "\n")
        for i in range(5):
            fh.write(json.dumps({"id": f"2405.{i:05d}", "categories": "cs.CL stat.ML", "title": f"new {i}",
                                 "abstract": ABSTRACT, "authors_parsed": [["Roe", "A", ""]],
                                 "versions": [{"created": "Wed, 1 May 2024 10:00:00 GMT"},
                                              {"created": "Fri, 3 May 2024 10:00:00 GMT"}]}) + "\n")
    chunks = list(iter_kaggle(str(path), cfg, chunk_rows=2))
    rows = pd.concat(chunks)
    assert len(rows) == 5
    assert rows["first_submitted"].iloc[0] == "2024-05-01"
    assert rows["last_updated"].iloc[0] == "2024-05-03"
    assert rows["arxiv_id"].iloc[0].endswith("v2")


def test_kaggle_source_writes_store(tmp_path):
    cfg = make_config(tmp_path)
    cfg.raw["data"].update(source="kaggle", allow_synthetic_fallback=False,
                           date_start="2024-01-01", date_end="2024-12-31",
                           kaggle_json=str(tmp_path / "snap.json"))
    with open(tmp_path / "snap.json", "w", encoding="utf-8") as fh:
        for i in range(40):
            fh.write(json.dumps({"id": f"2405.{i:05d}", "categories": "cs.CL", "title": f"paper {i}",
                                 "abstract": ABSTRACT, "authors_parsed": [["Roe", "A", ""]],
                                 "versions": [{"created": f"Wed, {1 + i % 28} May 2024 10:00:00 GMT"}]}) + "\n")
    df = build_corpus(cfg)
    assert len(df) == 40
    again = build_corpus(cfg)  # unchanged snapshot -> served from the store
    assert len(again) == 40


OAI_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
 <ListRecords>
  <record><header><identifier>oai:arXiv.org:2401.00001</identifier></header>
   <metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/">
    <id>2401.00001</id><created>2024-01-02</created><updated>2024-02-01</updated>
    <authors><author><keyname>Doe</keyname><forenames>Jane</forenames></author></authors>
    <title>A paper</title><categories>cs.LG stat.ML</categories>
    <doi>10.1/x</doi><license>http://creativecommons.org/licenses/by/4.0/</license>
    <abstract>Abstract text.</abstract>
   </arXiv></metadata></record>
  <record><header status="deleted"><identifier>oai:arXiv.org:2401.00002</identifier></header></record>
  {token}
 </ListRecords>
</OAI-PMH>"""


def test_oai_parse_page_and_resumption():
    recs, token = parse_page(OAI_PAGE.format(token="<resumptionToken>abc</resumptionToken>").encode())
    assert token == "abc"
    assert len(recs) == 1
    r = recs[0]
    assert r["authors"] == "Jane Doe" and r["primary_category"] == "cs.LG"
    assert r["first_submitted"] == "2024-01-02" and r["last_updated"] == "2024-02-01"


def test_oai_harvest_follows_tokens_and_records_state(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if "resumptionToken" in url:
            return OAI_PAGE.format(token="").encode()
        return OAI_PAGE.format(token="<resumptionToken>t1</resumptionToken>").encode()

    rows = harvest_range(date(2024, 1, 1), date(2024, 1, 31), endpoint="http://x/oai", set_spec="cs",
                         sleep_seconds=0, fetch=fetch)
    assert len(rows) == 2 and len(calls) == 2
    oai = {"endpoint": "http://x/oai", "set": "cs", "metadata_prefix": "arXiv",
           "sleep_seconds": 0, "max_retries": 0}
    state = tmp_path / "state.json"
    got = []
    n = harvest(date(2024, 1, 1), date(2024, 2, 29), oai, str(state),
                on_chunk=lambda r, lo, hi: got.append((lo, hi)), fetch=fetch)
    assert n == 4 and len(got) == 2
    assert json.loads(state.read_text())["completed_until"] == "2024-02-29"
    assert harvest(date(2024, 1, 1), date(2024, 2, 29), oai, str(state), fetch=fetch) == 0


def test_parse_atom_entry_keeps_primary_category():
    import xml.etree.ElementTree as ET

    xml = """<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <id>http://arxiv.org/abs/2501.01234v2</id><title>T</title><summary>S</summary>
      <published>2025-01-02T00:00:00Z</published><updated>2025-01-05T00:00:00Z</updated>
      <author><name>A</name></author>
      <category term="stat.ML"/><category term="cs.LG"/>
      <arxiv:primary_category term="cs.LG"/></entry>"""
    rec = parse_atom_entry(ET.fromstring(xml))
    assert rec["primary_category"] == "cs.LG"
    assert rec["arxiv_id"] == "2501.01234v2"
    assert rec["last_updated"].startswith("2025-01-05")
