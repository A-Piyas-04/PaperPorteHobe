"""Foundations and reading paths: the citation backbone per area (including pre-2023 works outside the
corpus), survey detection, and an ordered reading path per area. A newcomer's second question is
"what do I read first?", and a 2023+ corpus never shows the papers everything builds on."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger, save_json
from .backbone import BACKBONE_COLUMNS, backbone, reference_lists, subset_backbone
from .lookup import WorkLookup
from .path import reading_path
from .surveys import is_survey, survey_mask

__all__ = ["WorkLookup", "build_foundations", "reading_path", "backbone", "subset_backbone", "reference_lists",
           "is_survey", "survey_mask", "attach_reasons", "corpus_references", "write_empty"]

log = get_logger("foundations")
FOUNDATIONS_PARQUET = "foundations.parquet"
READING_PATHS_JSON = "reading_paths.json"


def corpus_references(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """The ``references`` column, or deterministic fake references for the synthetic corpus."""
    refs = df["references"] if "references" in df.columns else pd.Series([""] * len(df), index=df.index)
    if cfg["data"]["source"] == "synthetic" and not refs.fillna("").astype(str).str.strip().any():
        from .synthetic import synthetic_references

        return synthetic_references(df)
    return refs.fillna("").astype(str)


def attach_reasons(llm: Any, items: List[Dict], topic_of: Dict[int, str]) -> int:
    """Add why_read / what_you_learn / level to survey and foundation items that have an abstract.
    ``topic_of`` maps ``id(item)`` to its topic name. Returns the number of items enriched."""
    from ..llm import LLMError
    from . import prompts as _prompts  # noqa: F401  (registers "reading_reason")
    from .prompts import ReadingReason

    if llm is None or not llm.available():
        return 0
    todo = [it for it in items if it["step"] in ("survey", "foundation") and it.get("abstract")]
    if not todo:
        return 0
    keyed: Dict[str, List[Dict]] = {}
    for it in todo:
        keyed.setdefault(f"{it['paper_id']}|{it['step']}|{topic_of.get(id(it), '')}", []).append(it)
    req = [(k, {"topic": topic_of.get(id(v[0]), "this topic"), "step": v[0]["step"], "title": v[0]["title"],
                "abstract": v[0]["abstract"]}) for k, v in keyed.items()]
    try:
        results = llm.structured_many("reason", "reading_reason", ReadingReason, req)
    except LLMError as exc:
        log.info("Reading reasons skipped (%s); template reasons kept.", exc)
        return 0
    n = 0
    for k, res in results.items():
        if isinstance(res, LLMError) or res is None:
            continue
        for it in keyed.get(k, []):
            it.update(why_read=res.value.why_read, what_you_learn=res.value.what_you_learn, level=res.value.level)
            n += 1
    return n


def write_empty(out_dir: str) -> None:
    pd.DataFrame(columns=["area_id", *BACKBONE_COLUMNS]).to_parquet(os.path.join(out_dir, FOUNDATIONS_PARQUET),
                                                                    index=False)
    save_json({}, os.path.join(out_dir, READING_PATHS_JSON))


def build_foundations(cfg: Config, bundle_dir: str, llm: Optional[Any] = None) -> Dict:
    from ..artifacts import load_bundle

    f = cfg["foundations"]
    if not f["enabled"]:
        write_empty(bundle_dir)
        return {"enabled": False}
    b = load_bundle(cfg, directory=bundle_dir)
    assert isinstance(b.df.index, pd.RangeIndex), "reading_path needs a RangeIndex bundle frame"
    refs = corpus_references(b.df, cfg)
    synthetic = cfg["data"]["source"] == "synthetic"
    ref_lists = reference_lists(refs)
    bb = backbone(ref_lists, b.labels, int(f["min_citing_papers"]), float(f["min_share"]),
                  int(f["top_per_area"]), float(f["lift_epsilon"]))
    all_ids = sorted({w for t in bb.values() for w in t["openalex_id"]})
    works = WorkLookup(cfg, synthetic=synthetic).get(all_ids)
    problems_by_paper: Dict[str, int] = (b.problems["arxiv_id"].astype(str).value_counts().to_dict()
                                         if len(b.problems) else {})
    sizes = {k: int(v) for k, v in f["path_sizes"].items()}
    paths: Dict[str, List[Dict]] = {}
    topic_of: Dict[int, str] = {}
    from ..labeling import pretty_label

    for cid, info in sorted(b.clusters_meta.items()):
        idx = np.where(b.labels == int(cid))[0]
        if not len(idx):
            continue
        rows = bb.get(int(cid), pd.DataFrame(columns=BACKBONE_COLUMNS))
        n_base = int(rows["n_citing_base"].iloc[0]) if len(rows) else int(len(idx))
        items = reading_path(b.df, idx, b.embeddings, works, rows, problems_by_paper, sizes,
                             int(f["recent_months"]), int(f["frontier_months"]), n_base)
        name = pretty_label(info["keywords"]) if info.get("keywords") else info.get("label", f"Area {cid}")
        for it in items:
            topic_of[id(it)] = str(name)
        paths[str(int(cid))] = items
    n_reasons = 0
    if f["llm_reasons"]:
        from ..llm import get_llm

        n_reasons = attach_reasons(llm if llm is not None else get_llm(cfg),
                                   [it for items in paths.values() for it in items], topic_of)
    for items in paths.values():
        for it in items:
            it.pop("abstract", None)
    frames = []
    for cid, rows in bb.items():
        if len(rows):
            frames.append(rows.assign(area_id=int(cid)))
    fdf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["area_id", *BACKBONE_COLUMNS])
    if len(fdf) and len(works):
        meta_cols = ["openalex_id", "title", "year", "cited_by_count", "doi", "arxiv_id", "type", "venue", "url"]
        fdf = fdf.merge(works[meta_cols], on="openalex_id", how="left")
    fdf.to_parquet(os.path.join(bundle_dir, FOUNDATIONS_PARQUET), index=False)
    save_json(paths, os.path.join(bundle_dir, READING_PATHS_JSON))
    fitems = [it for items in paths.values() for it in items if it["step"] == "foundation"]
    with_refs = int(sum(1 for r in ref_lists if r))
    stats = {"enabled": True, "areas_with_path": int(sum(1 for v in paths.values() if v)),
             "mean_items": round(float(np.mean([len(v) for v in paths.values()])), 2) if paths else 0.0,
             "works_looked_up": int(len(works)), "works_missing": int(len(set(all_ids) - set(works["openalex_id"]))),
             "pre_2020_foundations": int(sum(1 for it in fitems if (it["year"] or 9999) < 2020)),
             "papers_with_references": with_refs, "reference_coverage": round(with_refs / max(1, len(b.df)), 4),
             "llm_reasons": n_reasons, "synthetic_references": bool(synthetic and not b.df.get(
                 "references", pd.Series(dtype=str)).fillna("").astype(str).str.strip().any())}
    log.info("Foundations: %s", stats)
    return stats
