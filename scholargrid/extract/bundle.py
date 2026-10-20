"""Turns stored extractions into the bundle tables the app and the gap engine read.

Only exploded tables ship; ``extractions.parquet`` (full payloads) never does. Quotes in
``problems.parquet`` are <= 300 chars.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..acronyms import mine as mine_acronyms
from ..config import Config
from ..utils import get_logger, l2_normalize, save_json
from .canonical import canonicalize_values
from .store import ExtractionStore

log = get_logger("extract.bundle")
FACETS = ("task", "method", "dataset", "metric", "domain", "language", "modality", "contribution")
FILES = ("paper_facets.parquet", "problems.parquet", "findings.parquet", "paper_profile.parquet")
PROBLEM_COLUMNS = ["problem_id", "arxiv_id", "statement", "quote", "section", "anchor", "section_kind",
                   "problem_type", "specificity", "verified_ratio", "source_mode"]
FINDING_COLUMNS = ["finding_id", "arxiv_id", "claim", "task", "dataset", "metric", "direction", "compared_to"]
PROFILE_COLUMNS = ["arxiv_id", "summary", "code_url", "data_released", "compute", "mode", "model"]


def explode(rows: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    facets: list = []
    problems: list = []
    findings: list = []
    profiles: list = []
    for r in rows.itertuples():
        p = json.loads(r.payload)
        aid = str(r.arxiv_id)
        for facet, key in (("task", "tasks"), ("dataset", "datasets"), ("metric", "metrics"),
                           ("domain", "domains"), ("language", "languages"), ("modality", "modalities"),
                           ("contribution", "contribution_types")):
            facets += [{"arxiv_id": aid, "facet": facet, "value_raw": v, "role": ""} for v in p.get(key, [])]
        facets += [{"arxiv_id": aid, "facet": "method", "value_raw": m["name"], "role": m["role"]}
                   for m in p.get("methods", [])]
        for q in json.loads(r.problems):
            pid = hashlib.sha1(f"{aid}|{q['quote']}".encode()).hexdigest()[:12]
            problems.append({"problem_id": pid, "arxiv_id": aid, **q, "source_mode": r.mode})
        for f in p.get("findings", []):
            fid = hashlib.sha1(f"{aid}|{f['claim']}".encode()).hexdigest()[:12]
            findings.append({"finding_id": fid, "arxiv_id": aid, **f})
        ev = p.get("evaluation") or {}
        profiles.append({"arxiv_id": aid, "summary": p.get("one_line_summary", ""), "code_url": p.get("code_url") or "",
                         "data_released": p.get("data_released"), "compute": p.get("compute", "unknown"),
                         "mode": r.mode, "model": r.model, **{f"eval_{k}": v for k, v in ev.items()}})
    return {"paper_facets": pd.DataFrame(facets, columns=["arxiv_id", "facet", "value_raw", "role"]),
            "problems": pd.DataFrame(problems, columns=PROBLEM_COLUMNS) if not problems else
            pd.DataFrame(problems).drop_duplicates("problem_id").reset_index(drop=True),
            "findings": pd.DataFrame(findings, columns=FINDING_COLUMNS) if not findings else
            pd.DataFrame(findings).drop_duplicates("finding_id").reset_index(drop=True),
            "paper_profile": pd.DataFrame(profiles, columns=PROFILE_COLUMNS) if not profiles else
            pd.DataFrame(profiles)}


def write_extraction_bundle(cfg: Config, df: pd.DataFrame, embedder: Any, out_dir: str) -> Dict:
    """Write the extraction tables + facet vocabulary (+ problem embeddings) into ``out_dir``."""
    rows = ExtractionStore(cfg).latest(df["arxiv_id"].astype(str)) if cfg["extract"]["enabled"] else pd.DataFrame()
    tables = explode(rows) if len(rows) else explode(pd.DataFrame(columns=["arxiv_id", "payload", "problems",
                                                                           "mode", "model"]))
    facets = tables["paper_facets"]
    vocab: Dict[str, Dict] = {}
    acr = mine_acronyms(df["abstract"].astype(str))
    e = cfg["extract"]
    facets["value"] = facets["value_raw"]
    for facet in FACETS:
        sub = facets[facets["facet"] == facet]
        if sub.empty:
            continue
        counts = Counter(sub["value_raw"])
        vecs = None
        if embedder is not None and facet in ("task", "method", "dataset", "metric", "domain") and len(counts) > 1:
            vecs = l2_normalize(np.asarray(embedder.transform(list(counts)), dtype=np.float32))
        mapping = canonicalize_values(counts, vecs, acr, float(e["canonical_min_cosine"]),
                                      int(e["canonical_min_string_ratio"]))
        facets.loc[sub.index, "value"] = sub["value_raw"].map(mapping)
        agg: Dict[str, Dict] = {}
        for raw, canon in mapping.items():
            a = agg.setdefault(canon, {"count": 0, "aliases": []})
            a["count"] += counts[raw]
            if raw != canon:
                a["aliases"].append(raw)
        vocab[facet] = dict(sorted(agg.items(), key=lambda kv: -kv[1]["count"]))
    tables["paper_facets"] = facets.drop_duplicates(["arxiv_id", "facet", "value", "role"]).reset_index(drop=True)
    for name, t in tables.items():
        t.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
    save_json(vocab, os.path.join(out_dir, "facet_vocab.json"))
    emb_path = os.path.join(out_dir, "problem_embeddings.npy")
    problems = tables["problems"]
    if embedder is not None and len(problems):
        vecs = l2_normalize(np.asarray(embedder.transform(problems["statement"].astype(str).tolist()),
                                       dtype=np.float32))
        np.save(emb_path, vecs.astype(np.float16))
    elif os.path.exists(emb_path):
        os.remove(emb_path)
    modes = {str(k): int(v) for k, v in rows["mode"].value_counts().items()} if len(rows) else {}
    stats = {"papers_with_extraction": int(len(rows)), "problems": int(len(problems)),
             "facet_values": {f: len(v) for f, v in vocab.items()}, "modes": modes}
    log.info("Extraction bundle: %s", stats)
    return stats
