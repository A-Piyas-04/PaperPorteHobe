"""Validation, robustness analysis and release gates (FR-15 .. FR-18).

* Cluster quality  : silhouette, DBCV (sampled), NPMI keyword coherence,
                     keyword diversity, noise fraction, subsample stability (ARI).
* Search quality   : graded relevance (0/1/2) -> nDCG@10, MRR, precision@10,
                     recall@25, for the configured mode and the dense / lexical
                     baselines. Hand-made judgments (``judgments_file``) take
                     precedence over the heuristic grades.
* Growth           : reliable / gated / unstable clusters, backtest hit rate.
* Projection check : robustness of sparse leads.
* Gates            : pass/fail per check with thresholds from config; the
                     publish stage refuses bundles that fail required gates.

Writes ``reports/validation_report.{json,md}`` and appends a line to
``reports/validation_history.jsonl`` so metrics can be tracked across releases.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .growth import backtest
from .search import search
from .utils import ensure_dir, get_logger, load_json, save_json

log = get_logger("validate")

# gate name -> (metric path in the report, comparison)
_GATES = {
    "search_ndcg_at_10": (("search", "ndcg_at_10"), ">="),
    "search_precision_at_10": (("search", "precision_at_10"), ">="),
    "search_recall_at_25": (("search", "recall_at_k"), ">="),
    "min_span_months": (("data", "span_months"), ">="),
    "max_noise_fraction": (("clusters", "noise_fraction"), "<="),
    "min_cluster_stability_ari": (("clusters", "stability", "mean_ari"), ">="),
    "min_ann_recall": (("index", "recall_at_25"), ">="),
    "min_openalex_match_rate": (("enrichment", "match_rate_older_3m"), ">="),
}


def validate(cfg: Config, df: pd.DataFrame, embeddings: np.ndarray, reduced: np.ndarray,
             labels: np.ndarray, clusters_meta: Dict, embedder, sparse_leads: list, *,
             bm25=None, index=None, quality: Optional[Dict] = None,
             growth: Optional[Dict] = None, stability: Optional[Dict] = None,
             ann_recall: Optional[float] = None, enrichment: Optional[Dict] = None,
             cluster_match: Optional[Dict] = None, write: bool = True) -> Dict:
    from .quality import span_months

    clusters = _validate_clusters(df, reduced, embeddings, labels, clusters_meta)
    clusters["stability"] = stability or {}
    report = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_version": cfg["pipeline_version"],
        "data": {"papers": int(len(df)), "span_months": span_months(df),
                 "quality": quality or {}},
        "clusters": clusters,
        "search": _validate_search(cfg, df, embeddings, labels, clusters_meta, embedder, bm25, index),
        "growth": _validate_growth(cfg, df, labels, growth),
        "projection_robustness": _validate_projection(sparse_leads, cfg),
        "index": {"backend": getattr(index, "backend", "exact"), "recall_at_25": ann_recall},
        "enrichment": enrichment or {},
        "cluster_match": cluster_match or {},
    }
    report["gates"] = evaluate_gates(cfg, report)
    if write:
        _write_markdown(cfg, report)
        save_json(report, os.path.join(cfg.reports_dir, "validation_report.json"))
        _append_history(cfg, report)
    return report


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def _lookup(report: Dict, path) -> Optional[float]:
    node = report
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, (int, float)) else None


def evaluate_gates(cfg: Config, report: Dict) -> Dict:
    v = cfg["validation"]
    required = set(v["required_gates"])
    results: List[Dict] = []
    for name, threshold in v["gates"].items():
        if name not in _GATES:
            log.warning("Unknown validation gate '%s' ignored.", name)
            continue
        path, op = _GATES[name]
        value = _lookup(report, path)
        passed = value is not None and (value >= threshold if op == ">=" else value <= threshold)
        results.append({"gate": name, "value": value, "threshold": threshold, "op": op,
                        "passed": bool(passed), "required": name in required})
    q = (report.get("data") or {}).get("quality") or {}
    results.append({"gate": "quality_checks", "value": q.get("failed", []), "threshold": "all pass",
                    "op": "==", "passed": bool(q.get("passed", False)),
                    "required": "quality_checks" in required})
    failed_required = [r["gate"] for r in results if r["required"] and not r["passed"]]
    return {"passed": not failed_required, "failed_required": failed_required,
            "enforced": bool(v["enforce_gates"]), "results": results}


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
def _validate_clusters(df, reduced, embeddings, labels, clusters_meta) -> Dict:
    cluster_ids = sorted({int(c) for c in labels if c != -1})
    cohesion = {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        centroid = np.asarray(embeddings[idx], dtype=np.float32).mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        cohesion[cid] = round(float((np.asarray(embeddings[idx], dtype=np.float32) @ centroid).mean()), 4)

    sil, dbcv = None, None
    clustered = labels != -1
    if len(cluster_ids) >= 2 and clustered.sum() > len(cluster_ids):
        from sklearn.metrics import silhouette_score

        try:
            sil = round(float(silhouette_score(reduced[clustered], labels[clustered],
                                               sample_size=min(5000, int(clustered.sum())),
                                               random_state=0)), 4)
        except Exception as exc:  # pragma: no cover
            log.warning("silhouette failed: %s", exc)
        dbcv = _dbcv(reduced, labels)

    keywords = {cid: (clusters_meta.get(cid) or {}).get("keywords", []) for cid in cluster_ids}
    all_kw = [k for kws in keywords.values() for k in kws]
    return {
        "n_clusters": len(cluster_ids),
        "n_noise": int((labels == -1).sum()),
        "noise_fraction": round(float((labels == -1).mean()), 4),
        "mean_embedding_cohesion": round(float(np.mean(list(cohesion.values()))), 4) if cohesion else None,
        "per_cluster_cohesion": cohesion,
        "silhouette_reduced": sil,
        "dbcv": dbcv,
        "npmi_coherence": _npmi(df, keywords),
        "keyword_diversity": round(len(set(all_kw)) / len(all_kw), 4) if all_kw else None,
    }


def _dbcv(reduced: np.ndarray, labels: np.ndarray, sample: int = 3000) -> Optional[float]:
    try:
        from hdbscan.validity import validity_index  # type: ignore
    except Exception:
        return None
    rng = np.random.default_rng(0)
    clusters = [c for c in np.unique(labels) if c != -1]
    if not clusters:
        return None
    # hdbscan's validity index breaks on clusters with < 2 sampled points, so sample per cluster.
    per = max(5, sample // len(clusters))
    parts = []
    for c in clusters:
        members = np.where(labels == c)[0]
        if len(members) >= 2:
            parts.append(rng.choice(members, size=min(per, len(members)), replace=False))
    if not parts:
        return None
    idx = np.concatenate(parts)
    sub_labels = labels[idx]
    if len(set(sub_labels.tolist())) < 2:
        return None
    try:
        return round(float(validity_index(np.asarray(reduced[idx], dtype=np.float64), sub_labels)), 4)
    except Exception as exc:  # pragma: no cover
        log.warning("DBCV failed: %s", exc)
        return None


def _npmi(df: pd.DataFrame, keywords: Dict[int, List[str]], top: int = 8) -> Optional[float]:
    """Mean pairwise NPMI of each cluster's top keywords over corpus documents."""
    from sklearn.feature_extraction.text import CountVectorizer

    vocab = sorted({k for kws in keywords.values() for k in kws[:top]})
    if len(vocab) < 2 or df.empty:
        return None
    texts = (df["title"].astype(str) + ". " + df["abstract"].astype(str)).tolist()
    vec = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), binary=True, lowercase=True)
    X = vec.fit_transform(texts).tocsc().astype(np.float32)
    n = X.shape[0]
    col = {t: i for i, t in enumerate(vocab)}
    p = np.asarray(X.sum(axis=0)).ravel() / n
    scores = []
    for kws in keywords.values():
        ids = [col[k] for k in kws[:top] if k in col]
        vals = []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                pij = float(X[:, i].multiply(X[:, j]).sum()) / n
                if pij <= 0 or p[i] <= 0 or p[j] <= 0:
                    vals.append(-1.0)
                    continue
                pmi = np.log(pij / (p[i] * p[j]))
                vals.append(float(pmi / -np.log(pij)) if pij < 1 else 1.0)
        if vals:
            scores.append(np.mean(vals))
    return round(float(np.mean(scores)), 4) if scores else None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def _grades(df: pd.DataFrame, q: Dict, text: pd.Series) -> np.ndarray:
    """Heuristic graded relevance over the whole corpus: +1 for an expected
    category, +1 for an expected keyword in the title or abstract."""
    exp_cats = tuple(q.get("expect_categories", []))
    kws = [w.lower() for w in q.get("expect_keywords", [])]
    cat_ok = (df["primary_category"].astype(str).str.startswith(exp_cats).to_numpy()
              if exp_cats else np.zeros(len(df), bool))
    kw_ok = np.zeros(len(df), dtype=bool)
    for w in kws:
        kw_ok |= text.str.contains(w, regex=False).to_numpy()
    return cat_ok.astype(int) + kw_ok.astype(int)


def _dcg(gains: Sequence[float]) -> float:
    return float(sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains)))


def search_metrics(retrieved: List[int], grades: np.ndarray, k: int, recall_k: int) -> Dict:
    top = [int(grades[i]) for i in retrieved[:k]]
    ideal = sorted(grades.tolist(), reverse=True)[:k]
    idcg = _dcg(ideal)
    first_rel = next((r for r, i in enumerate(retrieved, start=1) if grades[i] >= 1), None)
    pool = int((grades >= 2).sum())
    got = sum(1 for i in retrieved[:recall_k] if grades[i] >= 2)
    return {
        "ndcg_at_k": round(_dcg(top) / idcg, 4) if idcg > 0 else None,
        "precision_at_k": round(sum(1 for g in top if g >= 1) / max(1, len(top)), 4),
        "mrr": round(1.0 / first_rel, 4) if first_rel else 0.0,
        "recall_at_k": round(got / min(recall_k, pool), 4) if pool else None,
    }


def _judged_grades(df: pd.DataFrame, judged: Dict[str, int]) -> np.ndarray:
    m = df["arxiv_id"].astype(str).map(judged)
    return m.fillna(0).astype(int).to_numpy()


def _validate_search(cfg, df, embeddings, labels, clusters_meta, embedder, bm25, index) -> Dict:
    v = cfg["validation"]
    spec = load_json(cfg.abspath(v["queries_file"]))
    judgments: Dict[str, Dict[str, int]] = {}
    jf = v.get("judgments_file")
    if jf and os.path.exists(cfg.abspath(jf)):
        judgments = load_json(cfg.abspath(jf)).get("judgments", {})
    k, recall_k = int(v["search_top_k"]), int(v["recall_k"])
    modes = {"configured": None, "dense": "dense"}
    if bm25 is not None:
        modes["lexical"] = "lexical"
    per_mode: Dict[str, List[Dict]] = {m: [] for m in modes}
    text = (df["title"].astype(str) + " " + df["abstract"].astype(str)).str.lower()
    for q in spec["queries"]:
        judged = judgments.get(q["query"])
        grades = _judged_grades(df, judged) if judged else _grades(df, q, text)
        for name, mode in modes.items():
            res = search(q["query"], embedder, embeddings, df, labels, clusters_meta,
                         top_k=max(k, recall_k), bm25=bm25, index=index, cfg=cfg, mode=mode)
            m = search_metrics(res["result_indices"], grades, k, recall_k)
            m.update({"query": q["query"], "area": q.get("area", ""), "judged": bool(judged),
                      "mode": res["mode"]})
            per_mode[name].append(m)

    def mean(rows: List[Dict], key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    main = per_mode["configured"]
    return {
        "k": k, "recall_k": recall_k,
        "n_queries": len(main),
        "n_judged": sum(1 for r in main if r["judged"]),
        "mode": main[0]["mode"] if main else None,
        "ndcg_at_10": mean(main, "ndcg_at_k"),
        "precision_at_10": mean(main, "precision_at_k"),
        "mean_precision_at_k": mean(main, "precision_at_k"),
        "mrr": mean(main, "mrr"),
        "recall_at_k": mean(main, "recall_at_k"),
        "by_mode": {name: {"ndcg_at_10": mean(rows, "ndcg_at_k"),
                           "precision_at_10": mean(rows, "precision_at_k"),
                           "mrr": mean(rows, "mrr"), "recall_at_k": mean(rows, "recall_at_k")}
                    for name, rows in per_mode.items()},
        "per_query": [{"query": r["query"], "area": r["area"], "ndcg_at_10": r["ndcg_at_k"],
                       "precision_at_k": r["precision_at_k"], "mrr": r["mrr"],
                       "recall_at_k": r["recall_at_k"]} for r in main],
    }


# ---------------------------------------------------------------------------
# Growth / projection
# ---------------------------------------------------------------------------
def _validate_growth(cfg, df, labels, growth: Optional[Dict]) -> Dict:
    if growth is None:
        from .growth import compute_growth

        growth = compute_growth(df, labels, cfg)
    clusters = growth["clusters"]
    unstable = [cid for cid, c in clusters.items() if not c["stability"]["stable"]]
    gated = [cid for cid, c in clusters.items()
             if not c["windows"][str(growth["default_window"])]["passes_gate"]]
    return {
        "windows_months": growth["windows_months"],
        "cutoffs_months_back": cfg["validation"]["alternate_cutoffs_months"],
        "sufficient_history": growth["sufficient_history"],
        "history_days": growth["history_days"],
        "n_clusters": len(clusters),
        "n_reliable": growth.get("n_reliable", 0),
        "n_below_min_counts": len(gated),
        "n_unstable_clusters": len(unstable),
        "unstable_clusters": unstable[:50],
        "backtest": backtest(df, labels, cfg),
    }


def _validate_projection(sparse_leads: list, cfg: Config) -> Dict:
    if not sparse_leads:
        return {"n_leads": 0, "mean_robustness": None,
                "min_robustness_threshold": cfg["sparse"]["min_robustness"]}
    rob = [lead["robustness"] for lead in sparse_leads]
    return {
        "n_leads": len(sparse_leads),
        "mean_robustness": round(float(np.mean(rob)), 3),
        "min_robustness": round(float(np.min(rob)), 3),
        "min_robustness_threshold": cfg["sparse"]["min_robustness"],
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _append_history(cfg: Config, report: Dict) -> None:
    ensure_dir(cfg.reports_dir)
    s, c, g = report["search"], report["clusters"], report["growth"]
    line = {"at": report["generated_at"], "papers": report["data"]["papers"],
            "span_months": report["data"]["span_months"], "ndcg_at_10": s["ndcg_at_10"],
            "precision_at_10": s["precision_at_10"], "mrr": s["mrr"], "recall_at_k": s["recall_at_k"],
            "clusters": c["n_clusters"], "noise_fraction": c["noise_fraction"],
            "silhouette": c["silhouette_reduced"], "ari": (c.get("stability") or {}).get("mean_ari"),
            "reliable_growth": g["n_reliable"], "gates_passed": report["gates"]["passed"]}
    with open(os.path.join(cfg.reports_dir, "validation_history.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _write_markdown(cfg: Config, report: Dict) -> None:
    ensure_dir(cfg.reports_dir)
    c, s, g, p = report["clusters"], report["search"], report["growth"], report["projection_robustness"]
    gates = report["gates"]
    st = c.get("stability") or {}
    bt = g.get("backtest") or {}
    lines = [
        "# ScholarGrid — Validation Report",
        "",
        f"Pipeline version: `{cfg['pipeline_version']}` · generated {report['generated_at']}",
        "",
        f"## Release gates: {'PASS' if gates['passed'] else 'FAIL'}"
        f"{' (enforced)' if gates['enforced'] else ' (advisory)'}",
        "",
        "| Gate | Value | Threshold | Required | Result |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r['gate']} | {_fmt(r['value'])} | {r['op']} {r['threshold']} | "
              f"{'yes' if r['required'] else 'no'} | {'pass' if r['passed'] else 'FAIL'} |"
              for r in gates["results"]]
    lines += [
        "",
        "## Data",
        f"- Papers: **{report['data']['papers']:,}**, span: **{report['data']['span_months']} months**",
        "",
        "## Cluster validation (FR-15)",
        f"- Clusters: **{c['n_clusters']}**, noise: {c['n_noise']} ({c['noise_fraction']:.1%})",
        f"- Silhouette (reduced space): **{_fmt(c['silhouette_reduced'])}** · DBCV: **{_fmt(c['dbcv'])}**",
        f"- NPMI keyword coherence: **{_fmt(c['npmi_coherence'])}** · keyword diversity: "
        f"**{_fmt(c['keyword_diversity'])}**",
        f"- Subsample stability (ARI over {st.get('runs', 0)} runs): **{_fmt(st.get('mean_ari'))}**, "
        f"clusters surviving < 50% of runs: {len(st.get('unstable_clusters', []))}",
        "",
        "## Search validation (FR-16)",
        f"- Queries: **{s['n_queries']}** ({s['n_judged']} with hand judgments), mode: `{s['mode']}`",
        f"- nDCG@{s['k']}: **{_fmt(s['ndcg_at_10'])}** · precision@{s['k']}: **{_fmt(s['precision_at_10'])}** "
        f"· MRR: **{_fmt(s['mrr'])}** · recall@{s['recall_k']}: **{_fmt(s['recall_at_k'])}**",
        "",
        "| Mode | nDCG@10 | precision@10 | MRR | recall@25 |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {m} | {_fmt(v['ndcg_at_10'])} | {_fmt(v['precision_at_10'])} | {_fmt(v['mrr'])} | "
              f"{_fmt(v['recall_at_k'])} |" for m, v in s["by_mode"].items()]
    worst = sorted([q for q in s["per_query"] if q["ndcg_at_10"] is not None],
                   key=lambda q: q["ndcg_at_10"])[:10]
    if worst:
        lines += ["", "Weakest queries:", "", "| Query | nDCG@10 | precision@10 |", "|---|---|---|"]
        lines += [f"| {q['query']} | {_fmt(q['ndcg_at_10'])} | {_fmt(q['precision_at_k'])} |" for q in worst]
    lines += [
        "",
        "## Growth validation (FR-17)",
        f"- History: {g['history_days']} days; sufficient for the default window: "
        f"**{'yes' if g['sufficient_history'] else 'no'}**",
        f"- Clusters with a reliable growth signal: **{g['n_reliable']} / {g['n_clusters']}** "
        f"(below minimum counts: {g['n_below_min_counts']}, unstable: {g['n_unstable_clusters']})",
        "- Backtest: " + (f"{bt.get('flagged_growing')} flagged growing as of {bt.get('reference_date')}, "
                           f"hit rate **{_fmt(bt.get('hit_rate'))}**" if bt.get("available")
                           else f"not available ({bt.get('reason')})"),
        "",
        "## Index",
        f"- Backend: `{report['index']['backend']}`, ANN recall@25: **{_fmt(report['index']['recall_at_25'])}**",
        "",
        "## Projection robustness (FR-18)",
        f"- Sparse leads reported: **{p['n_leads']}**",
        f"- Mean robustness: **{_fmt(p.get('mean_robustness'))}** (threshold {p['min_robustness_threshold']})",
        "",
        "> Reminder: sparsity is not proof of novelty; growth is not research quality; "
        "the 2D map is a visualisation, not evidence by itself.",
    ]
    with open(os.path.join(cfg.reports_dir, "validation_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
