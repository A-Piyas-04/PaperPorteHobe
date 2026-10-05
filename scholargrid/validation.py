"""Validation & robustness analysis (FR-15 .. FR-18).

* Cluster quality  : sizes, embedding cohesion, silhouette.
* Search quality   : relevance rate / precision@k on a known-item benchmark.
* Growth stability : recompute across windows and alternate cutoff dates.
* Projection check : fraction of sparse leads robust across projections.

Writes a machine-readable JSON and a human-readable Markdown report that the app
surfaces (FR-13 / FR-14 / NFR-06).
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from .embeddings import Embedder
from .growth import compute_growth
from .search import search
from .utils import ensure_dir, get_logger, load_json, save_json

log = get_logger("validate")


def validate(cfg: Config, df: pd.DataFrame, embeddings: np.ndarray, reduced: np.ndarray,
             labels: np.ndarray, clusters_meta: Dict, embedder: Embedder,
             sparse_leads: list) -> Dict:
    report = {
        "clusters": _validate_clusters(reduced, embeddings, labels),
        "search": _validate_search(cfg, df, embeddings, labels, clusters_meta, embedder),
        "growth": _validate_growth(cfg, df, labels),
        "projection_robustness": _validate_projection(sparse_leads, cfg),
    }
    _write_markdown(cfg, report)
    save_json(report, os.path.join(cfg.reports_dir, "validation_report.json"))
    return report


def _validate_clusters(reduced: np.ndarray, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    cluster_ids = sorted({int(c) for c in labels if c != -1})
    cohesion = {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        centroid = embeddings[idx].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        cohesion[cid] = round(float((embeddings[idx] @ centroid).mean()), 4)

    sil = None
    clustered = labels != -1
    if len(cluster_ids) >= 2 and clustered.sum() > len(cluster_ids):
        try:
            from sklearn.metrics import silhouette_score

            sil = round(float(silhouette_score(reduced[clustered], labels[clustered])), 4)
        except Exception as exc:  # pragma: no cover
            log.warning("silhouette failed: %s", exc)
    return {
        "n_clusters": len(cluster_ids),
        "n_noise": int((labels == -1).sum()),
        "noise_fraction": round(float((labels == -1).mean()), 4),
        "mean_embedding_cohesion": round(float(np.mean(list(cohesion.values()))), 4) if cohesion else None,
        "per_cluster_cohesion": cohesion,
        "silhouette_reduced": sil,
    }


def _validate_search(cfg, df, embeddings, labels, clusters_meta, embedder) -> Dict:
    spec = load_json(cfg.abspath(cfg["validation"]["queries_file"]))
    k = int(cfg["validation"]["search_top_k"])
    per_query, rates = [], []
    for q in spec["queries"]:
        res = search(q["query"], embedder, embeddings, df, labels, clusters_meta, top_k=k)
        exp_cats = set(q.get("expect_categories", []))
        exp_kw = [w.lower() for w in q.get("expect_keywords", [])]
        hits = 0
        for r in res["results"]:
            cat_ok = any(r["primary_category"].startswith(c) for c in exp_cats) if exp_cats else False
            text = (r["title"]).lower()
            kw_ok = any(w in text for w in exp_kw)
            if cat_ok or kw_ok:
                hits += 1
        rate = hits / max(1, len(res["results"]))
        rates.append(rate)
        per_query.append({"query": q["query"], "precision_at_k": round(rate, 3),
                          "n_results": len(res["results"])})
    return {
        "k": k,
        "n_queries": len(per_query),
        "mean_precision_at_k": round(float(np.mean(rates)), 4) if rates else None,
        "per_query": per_query,
    }


def _validate_growth(cfg, df, labels) -> Dict:
    windows = list(cfg["growth"]["windows_months"])
    cutoffs = ["auto"] + list(cfg["validation"]["alternate_cutoffs"])
    # Collect default-window relative growth per cluster across cutoffs.
    series: Dict[int, list] = {}
    for cutoff in cutoffs:
        g = compute_growth(df, labels, cfg, reference_date=cutoff, windows=windows)
        for cid, info in g["clusters"].items():
            series.setdefault(cid, []).append(info["relative_growth_default"])
    unstable = []
    for cid, vals in series.items():
        arr = np.array(vals, dtype=float)
        mean = arr.mean()
        cv = arr.std() / mean if mean > 0 else float("inf")
        if not np.isfinite(cv) or cv > 0.5:
            unstable.append({"cluster_id": cid, "values": [round(v, 3) for v in vals],
                             "coef_variation": round(cv, 3) if np.isfinite(cv) else None})
    return {
        "windows_months": windows,
        "cutoffs_tested": cutoffs,
        "n_clusters": len(series),
        "n_unstable_clusters": len(unstable),
        "unstable_clusters": unstable,
    }


def _validate_projection(sparse_leads: list, cfg: Config) -> Dict:
    if not sparse_leads:
        return {"n_leads": 0, "mean_robustness": None, "min_robustness_threshold": cfg["sparse"]["min_robustness"]}
    rob = [lead["robustness"] for lead in sparse_leads]
    return {
        "n_leads": len(sparse_leads),
        "mean_robustness": round(float(np.mean(rob)), 3),
        "min_robustness": round(float(np.min(rob)), 3),
        "min_robustness_threshold": cfg["sparse"]["min_robustness"],
    }


def _write_markdown(cfg: Config, report: Dict) -> None:
    ensure_dir(cfg.reports_dir)
    c, s, g, p = report["clusters"], report["search"], report["growth"], report["projection_robustness"]
    lines = [
        "# ScholarGrid — Validation Report",
        "",
        f"Pipeline version: `{cfg['pipeline_version']}`",
        "",
        "## Cluster validation (FR-15)",
        f"- Clusters: **{c['n_clusters']}**, noise: {c['n_noise']} ({c['noise_fraction']:.1%})",
        f"- Mean embedding cohesion: **{c['mean_embedding_cohesion']}**",
        f"- Silhouette (reduced space): **{c['silhouette_reduced']}**",
        "",
        "## Search validation (FR-16)",
        f"- Benchmark queries: **{s['n_queries']}**, k = {s['k']}",
        f"- Mean precision@k (relevance rate): **{s['mean_precision_at_k']}**",
        "",
        "| Query | precision@k |",
        "|---|---|",
    ]
    lines += [f"| {q['query']} | {q['precision_at_k']} |" for q in s["per_query"]]
    lines += [
        "",
        "## Growth validation (FR-17)",
        f"- Windows tested (months): {g['windows_months']}",
        f"- Cutoffs tested: {g['cutoffs_tested']}",
        f"- Unstable clusters (CV>0.5 across cutoffs): **{g['n_unstable_clusters']} / {g['n_clusters']}**",
        "",
        "## Projection robustness (FR-18)",
        f"- Sparse leads reported: **{p['n_leads']}**",
        f"- Mean robustness: **{p.get('mean_robustness')}** (threshold {p['min_robustness_threshold']})",
        "",
        "> Reminder: sparsity is not proof of novelty; growth is not research quality; "
        "the 2D map is a visualisation, not evidence by itself.",
    ]
    with open(os.path.join(cfg.reports_dir, "validation_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
