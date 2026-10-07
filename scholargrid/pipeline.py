"""Staged offline pipeline.

    ingest -> enrich -> embed -> analyze -> validate -> publish

``analyze`` covers index -> reduce -> cluster -> match IDs -> stability ->
label -> project -> growth -> sparse -> graph -> BM25 and writes the bundle.
Every stage writes its outputs under ``paths.stages_dir`` (or the staging
bundle), so any stage can be re-run alone from cached inputs:

    python -m scholargrid.pipeline                       # everything
    python -m scholargrid.pipeline --stage embed         # one stage
    python -m scholargrid.pipeline --from-stage analyze  # analyze, validate, publish
    python -m scholargrid.pipeline --config configs/production.yaml
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import __version__
from .config import Config, capabilities, load_config
from .utils import configure_logging, ensure_dir, get_logger, load_json, save_json, set_seed, stage_timer

log = get_logger("pipeline")

STAGES = ["ingest", "enrich", "embed", "analyze", "validate", "publish"]
ProgressFn = Optional[Callable[[float, str], None]]
_PROGRESS = {"ingest": (0.05, "Ingesting arXiv corpus…"), "enrich": (0.15, "Looking up citations…"),
             "embed": (0.25, "Embedding papers…"), "analyze": (0.45, "Clustering and mapping…"),
             "validate": (0.9, "Validating…"), "publish": (0.97, "Publishing release…")}


class Context:
    """In-memory hand-off between stages, with on-disk fallbacks."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = ensure_dir(cfg.stages_dir)
        self.timings: Dict[str, Dict] = {}
        self.data: Dict[str, Any] = {}

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def save_frame(self, key: str, df: pd.DataFrame) -> None:
        self.data[key] = df
        df.to_parquet(self.path(f"{key}.parquet"), index=False)
        save_json(dict(df.attrs), self.path(f"{key}.attrs.json"))

    def frame(self, key: str) -> pd.DataFrame:
        if key in self.data:
            return self.data[key]
        path = self.path(f"{key}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stage output '{key}' missing; run the earlier stage first ({path}).")
        df = pd.read_parquet(path)
        attrs_path = self.path(f"{key}.attrs.json")
        if os.path.exists(attrs_path):
            df.attrs.update(load_json(attrs_path))
        self.data[key] = df
        return df

    def save_json(self, key: str, obj: Any) -> None:
        self.data[key] = obj
        save_json(obj, self.path(f"{key}.json"))

    def json(self, key: str, default: Any = None) -> Any:
        if key in self.data:
            return self.data[key]
        path = self.path(f"{key}.json")
        return load_json(path) if os.path.exists(path) else default


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_ingest(ctx: Context) -> None:
    from .data_ingest import build_corpus
    from .quality import run_checks
    from .release import current_release

    cfg = ctx.cfg
    with stage_timer("ingest", ctx.timings) as info:
        df = build_corpus(cfg)
        prev_count = None
        cur = current_release(cfg)
        if cur:
            manifest = os.path.join(cfg.releases_dir, cur, "manifest.json")
            if os.path.exists(manifest):
                prev_count = load_json(manifest).get("corpus", {}).get("n_papers")
        ctx.save_json("quality", run_checks(df, cfg, previous_count=prev_count))
        ctx.save_frame("corpus", df)
        info.update(papers=len(df), available=df.attrs.get("available"))


def stage_enrich(ctx: Context) -> None:
    from .enrich import enrich, match_stats

    with stage_timer("enrich", ctx.timings) as info:
        df = enrich(ctx.frame("corpus"), ctx.cfg)
        stats = match_stats(df)
        ctx.save_json("enrichment", stats)
        ctx.save_frame("enriched", df)
        info.update(papers=len(df), matched=stats["matched"])


def stage_embed(ctx: Context) -> None:
    from .embeddings import Embedder, paper_text

    cfg = ctx.cfg
    df = ctx.frame("enriched")
    backend = cfg.resolve_embedding_backend()
    with stage_timer("embed", ctx.timings, backend=backend) as info:
        embedder = Embedder(backend, cfg)
        vecs = embedder.fit_transform(paper_text(df), ids=df["arxiv_id"].astype(str).tolist())
        np.save(ctx.path("embeddings.npy"), vecs.astype(np.float32))
        embedder.save(ctx.path("embedder"))
        ctx.data["embeddings"], ctx.data["embedder"] = vecs, embedder
        ctx.save_json("embed_meta", {"backend": backend, "fingerprint": embedder.fingerprint,
                                     "n": int(len(vecs)), "dim": int(vecs.shape[1])})
        info.update(papers=len(vecs), dim=int(vecs.shape[1]))


def _previous_assignments(cfg: Config) -> Optional[pd.DataFrame]:
    from .artifacts import META_JSON, load_papers
    from .release import resolve_bundle_dir

    try:
        prev_dir = resolve_bundle_dir(cfg)
    except FileNotFoundError:
        return None
    if not os.path.exists(os.path.join(prev_dir, META_JSON)):
        return None
    try:
        return load_papers(prev_dir)[["arxiv_id", "cluster_id"]]
    except Exception as exc:  # pragma: no cover - corrupt previous bundle
        log.warning("Could not read previous cluster assignments (%s).", exc)
        return None


def stage_analyze(ctx: Context) -> None:
    from .artifacts import save_bundle
    from .bm25 import BM25Index
    from .cluster_match import match_clusters
    from .clustering import cluster, cluster_stability
    from .embeddings import Embedder, paper_text
    from .graph import build_graph
    from .growth import compute_growth
    from .index import NeighborIndex
    from .labeling import generate_labels
    from .reduce import project_2d, reduce_analytical
    from .sparse import detect_sparse
    from .utils import l2_normalize

    cfg = ctx.cfg
    df = ctx.frame("enriched")
    embeddings = ctx.data.get("embeddings")
    if embeddings is None:
        embeddings = np.load(ctx.path("embeddings.npy"))
    embedder = ctx.data.get("embedder") or Embedder.load(ctx.path("embedder"), cfg, lazy=True)
    embeddings = l2_normalize(embeddings)
    t = ctx.timings

    with stage_timer("index", t) as info:
        index = NeighborIndex.build(embeddings, cfg)
        recall = index.recall_at_k(25, int(cfg["index"]["recall_sample"]), cfg["seed"])
        info.update(backend=index.backend, recall_at_25=recall)
    with stage_timer("reduce", t) as info:
        reduced, reduce_backend = reduce_analytical(embeddings, cfg)
        info.update(backend=reduce_backend)
    with stage_timer("cluster", t) as info:
        labels, cluster_backend = cluster(reduced, cfg)
        match: Dict[str, Any] = {"report": {}}
        if cfg["cluster"]["match_previous"]:
            match = match_clusters(df["arxiv_id"], labels, embeddings, _previous_assignments(cfg),
                                   float(cfg["cluster"]["match_min_score"]))
            labels = match["labels"]
        stability = cluster_stability(reduced, labels, cfg, cluster_backend)
        info.update(backend=cluster_backend, clusters=len({int(c) for c in labels if c != -1}),
                    ari=stability.get("mean_ari"))
    with stage_timer("label", t):
        clusters_meta = generate_labels(df, embeddings, labels, cfg)
        for cid, surv in stability.get("per_cluster_survival", {}).items():
            if int(cid) in clusters_meta:
                clusters_meta[int(cid)]["stability_survival"] = surv
    with stage_timer("project", t) as info:
        coords2d, landscape_backend = project_2d(embeddings, cfg)
        info.update(backend=landscape_backend)
    with stage_timer("growth", t) as info:
        growth = compute_growth(df, labels, cfg)
        for cid, g in growth["clusters"].items():
            if cid in clusters_meta:
                clusters_meta[cid]["growth"] = g
        info.update(reliable=growth["n_reliable"], sufficient=growth["sufficient_history"])
    with stage_timer("sparse", t) as info:
        sparse_leads = detect_sparse(df, embeddings, coords2d, labels, cfg, clusters_meta)
        for lead in sparse_leads:
            g = growth["clusters"].get(lead.get("nearest_cluster_id"))
            if g:
                w = g["windows"][str(growth["default_window"])]
                lead["nearest_cluster_growth"] = {"relative_growth": w["relative_growth"], "ci": w["ci"],
                                                  "reliable": g["reliable"],
                                                  "recent_count": w["recent_count"]}
        info.update(leads=len(sparse_leads))
    with stage_timer("graph", t) as info:
        edges, neighbors = build_graph(df, embeddings, cfg, index=index)
        info.update(edges=len(edges))
    with stage_timer("bm25", t) as info:
        bm25 = BM25Index.build(paper_text(df))
        info.update(vocab=len(bm25.vocab))

    caps = capabilities()
    meta = {
        "pipeline_version": cfg["pipeline_version"],
        "scholargrid_version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": cfg["environment"],
        "config_hash": cfg.config_hash(),
        "seed": cfg["seed"],
        "snapshot": _snapshot(df, cfg),
        "embedding_model": embedder.fingerprint,
        "backends": {"available": caps, "embedding": embedder.backend, "reduce": reduce_backend,
                     "cluster": cluster_backend, "landscape": landscape_backend,
                     "index": index.backend},
        "params": {k: cfg[k] for k in ("embedding", "index", "reduce", "cluster", "landscape",
                                       "growth", "sparse", "search")},
        "counts": {
            "papers": int(len(df)), "clusters": len(clusters_meta),
            "noise": int((labels == -1).sum()), "sparse_leads": len(sparse_leads),
            "embedding_dim": int(embeddings.shape[1]), "edges": len(edges),
            "with_authors": int((df["authors"].fillna("") != "").sum()),
            "with_references": int((df["reference_count"] > 0).sum()),
            "with_citations": int((df["cited_by_count"] > 0).sum()),
        },
        "quality": ctx.json("quality", {}),
        "enrichment": ctx.json("enrichment", {}),
        "cluster_match": match["report"],
        "cluster_stability": {k: v for k, v in stability.items() if k != "per_cluster_survival"},
        "ann_recall_at_25": recall,
        "sparse_gate": {"enabled": len(df) >= int(cfg["sparse"]["min_corpus_size"]),
                        "min_corpus_size": int(cfg["sparse"]["min_corpus_size"])},
        "stage_timings": dict(t),
    }
    with stage_timer("save", t):
        save_bundle(cfg, df, embeddings, reduced, coords2d, labels, clusters_meta, growth,
                    sparse_leads, embedder, meta, edges=edges, neighbors=neighbors,
                    bm25=bm25, index=index)
    ctx.data.update(df=df, embeddings=embeddings, reduced=reduced, labels=labels,
                    clusters_meta=clusters_meta, growth=growth, sparse_leads=sparse_leads,
                    embedder=embedder, index=index, bm25=bm25, meta=meta, stability=stability,
                    ann_recall=recall)


def _snapshot(df: pd.DataFrame, cfg: Config) -> Dict:
    from .data_ingest import snapshot_info

    return snapshot_info(df, cfg)


def stage_validate(ctx: Context) -> None:
    from .validation import validate

    cfg = ctx.cfg
    d = ctx.data
    if "labels" not in d:
        from .artifacts import load_bundle

        b = load_bundle(cfg, directory=cfg.processed_dir)
        d.update(df=b.df, embeddings=b.embeddings, labels=b.labels, clusters_meta=b.clusters_meta,
                 growth=b.growth, sparse_leads=b.sparse_leads, embedder=b.embedder,
                 index=b.index, bm25=b.bm25, meta=b.meta,
                 reduced=np.load(os.path.join(cfg.processed_dir, "reduced.npy")),
                 stability=b.meta.get("cluster_stability", {}),
                 ann_recall=b.meta.get("ann_recall_at_25"))
    with stage_timer("validate", ctx.timings) as info:
        report = validate(cfg, d["df"], d["embeddings"], d["reduced"], d["labels"], d["clusters_meta"],
                          d["embedder"], d["sparse_leads"], bm25=d["bm25"], index=d["index"],
                          quality=d["meta"].get("quality"), growth=d["growth"],
                          stability=d["stability"], ann_recall=d["ann_recall"],
                          enrichment=d["meta"].get("enrichment"),
                          cluster_match=d["meta"].get("cluster_match"))
        d["report"] = report
        info.update(gates_passed=report["gates"]["passed"], ndcg=report["search"]["ndcg_at_10"])


def stage_publish(ctx: Context) -> None:
    from .release import publish

    cfg = ctx.cfg
    meta = ctx.data.get("meta") or load_json(os.path.join(cfg.processed_dir, "meta.json"))
    report = ctx.data.get("report") or load_json(os.path.join(cfg.reports_dir, "validation_report.json"))
    with stage_timer("publish", ctx.timings) as info:
        meta = dict(meta)
        meta["stage_timings"] = {**meta.get("stage_timings", {}), **ctx.timings}
        save_json(meta, os.path.join(cfg.processed_dir, "meta.json"))
        rid = publish(cfg, meta, report, force=bool(ctx.data.get("force_publish")))
        ctx.data["release"] = rid
        info.update(release=rid)


_FUNCS = {"ingest": stage_ingest, "enrich": stage_enrich, "embed": stage_embed,
          "analyze": stage_analyze, "validate": stage_validate, "publish": stage_publish}


def run(cfg: Config, progress: ProgressFn = None, stages: Optional[List[str]] = None,
        force_publish: bool = False) -> Tuple[Dict, Dict]:
    """Run ``stages`` (default: all). Returns ``(meta, validation_report)``."""
    configure_logging(cfg["monitoring"]["json_logs"])
    set_seed(cfg["seed"])
    stages = stages or STAGES
    ctx = Context(cfg)
    ctx.data["force_publish"] = force_publish
    t0 = time.time()
    log.info("ScholarGrid v%s | env=%s | stages=%s | backends available: %s",
             __version__, cfg["environment"], stages, capabilities())
    for name in stages:
        frac, msg = _PROGRESS[name]
        if progress is not None:
            progress(frac, msg)
        _FUNCS[name](ctx)
    if progress is not None:
        progress(1.0, "Done.")
    meta = ctx.data.get("meta") or {}
    report = ctx.data.get("report") or {}
    log.info("Pipeline finished in %.1fs (stages=%s, release=%s)", time.time() - t0, stages,
             ctx.data.get("release"))
    return meta, report


def main(argv: Optional[List[str]] = None) -> None:
    from .monitoring import capture_exception, init_error_tracking

    p = argparse.ArgumentParser(description="ScholarGrid offline pipeline")
    p.add_argument("--config", default=None, help="path to a config YAML")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--stage", choices=STAGES, help="run a single stage")
    group.add_argument("--from-stage", choices=STAGES, help="run this stage and the ones after it")
    p.add_argument("--no-publish", action="store_true", help="skip the publish stage")
    p.add_argument("--force-publish", action="store_true",
                   help="publish even if required validation gates fail (recorded in the manifest)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    init_error_tracking("pipeline", cfg["pipeline_version"])
    if args.stage:
        stages = [args.stage]
    elif args.from_stage:
        stages = STAGES[STAGES.index(args.from_stage):]
    else:
        stages = list(STAGES)
    if args.no_publish and "publish" in stages:
        stages.remove("publish")
    try:
        meta, report = run(cfg, stages=stages, force_publish=args.force_publish)
    except Exception as exc:
        capture_exception(exc)
        raise
    if report:
        g = report["gates"]
        log.info("search nDCG@10=%s precision@10=%s | clusters=%s | gates %s",
                 report["search"]["ndcg_at_10"], report["search"]["precision_at_10"],
                 report["clusters"]["n_clusters"], "PASSED" if g["passed"] else f"FAILED {g['failed_required']}")


if __name__ == "__main__":
    main()
