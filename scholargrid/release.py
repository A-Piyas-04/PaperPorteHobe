"""Versioned data releases, manifests, changelogs and rollback.

A run writes its bundle to ``paths.processed_dir`` (staging). ``publish``
checks the validation gates, copies the bundle to
``paths.releases_dir/<YYYY.MM.DD>`` with a ``manifest.json`` and a user-facing
``changelog.json``, points ``CURRENT`` at it and prunes old releases (keeping
``release.keep_last`` plus any pinned one). The app serves ``release.pin`` (or
``SCHOLARGRID_RELEASE``) if set, else ``CURRENT``, else the staging bundle, so
rollback is a one-line config change or ``python -m scholargrid.release rollback <id>``.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
from datetime import UTC, datetime
from typing import Dict, List, Optional

import pandas as pd

from .config import Config, load_config
from .utils import get_logger, load_json, save_json

log = get_logger("release")

CURRENT_FILE = "CURRENT"
_LIBS = ["numpy", "pandas", "scikit-learn", "scipy", "sentence-transformers", "torch",
         "umap-learn", "hdbscan", "faiss-cpu", "duckdb", "streamlit"]


class PublishError(RuntimeError):
    pass


def library_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for lib in _LIBS:
        try:
            out[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            out[lib] = None
    return out


def list_releases(cfg: Config) -> List[str]:
    root = cfg.releases_dir
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "manifest.json")))


def current_release(cfg: Config) -> Optional[str]:
    path = os.path.join(cfg.releases_dir, CURRENT_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        rid = fh.read().strip()
    return rid or None


def resolve_bundle_dir(cfg: Config) -> str:
    pin = cfg["release"].get("pin")
    if pin:
        path = os.path.join(cfg.releases_dir, str(pin))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Pinned release '{pin}' not found in {cfg.releases_dir}")
        return path
    if cfg["release"]["enabled"]:
        cur = current_release(cfg)
        if cur and os.path.isdir(os.path.join(cfg.releases_dir, cur)):
            return os.path.join(cfg.releases_dir, cur)
    return cfg.processed_dir


def _new_id(cfg: Config) -> str:
    base = datetime.now(UTC).strftime("%Y.%m.%d")
    existing = set(list_releases(cfg))
    if base not in existing:
        return base
    n = 2
    while f"{base}.{n}" in existing:
        n += 1
    return f"{base}.{n}"


def _rel(path: str) -> str:
    from .config import REPO_ROOT

    try:
        return os.path.relpath(path, start=REPO_ROOT)
    except ValueError:
        return path


def build_manifest(cfg: Config, release_id: str, meta: Dict, report: Dict) -> Dict:
    snap = meta.get("snapshot", {})
    return {
        "release": release_id,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": cfg["environment"],
        "config_path": _rel(cfg.path),
        "config_hash": cfg.config_hash(),
        "pipeline_version": cfg["pipeline_version"],
        "corpus": {k: snap.get(k) for k in ("source", "profile", "sampling", "n_papers", "n_available",
                                            "earliest_included_date", "latest_included_date",
                                            "span_months", "n_categories", "harvested_at")},
        "backends": meta.get("backends", {}),
        "embedding_model": meta.get("embedding_model"),
        "libraries": library_versions(),
        "quality": (report.get("data") or {}).get("quality", {}),
        "validation": {
            "gates": report.get("gates", {}),
            "search": {k: (report.get("search") or {}).get(k)
                       for k in ("n_queries", "ndcg_at_10", "precision_at_10", "mrr", "recall_at_k")},
            "clusters": {k: (report.get("clusters") or {}).get(k)
                         for k in ("n_clusters", "noise_fraction", "silhouette_reduced", "dbcv")},
            "growth_reliable": (report.get("growth") or {}).get("n_reliable"),
        },
        "enrichment": {k: v for k, v in (meta.get("enrichment") or {}).items()
                       if k not in ("by_month", "by_category")},
        "cluster_match": meta.get("cluster_match", {}),
    }


def build_changelog(cfg: Config, staging: str, previous: Optional[str], meta: Dict) -> Dict:
    from .artifacts import PAPERS_PARQUET, load_papers

    now = load_papers(staging)
    summary: Dict = {"papers": int(len(now)), "areas": int(meta.get("counts", {}).get("clusters", 0))}
    if not previous:
        summary.update({"first_release": True})
        return summary
    prev_dir = os.path.join(cfg.releases_dir, previous)
    prev = load_papers(prev_dir) if os.path.exists(os.path.join(prev_dir, PAPERS_PARQUET)) or \
        os.path.exists(os.path.join(prev_dir, "papers.csv")) else pd.DataFrame(columns=["arxiv_id"])
    new_ids, old_ids = set(now["arxiv_id"]), set(prev["arxiv_id"])
    match = meta.get("cluster_match", {})
    summary.update({
        "previous_release": previous,
        "new_papers": len(new_ids - old_ids),
        "removed_papers": len(old_ids - new_ids),
        "new_areas": match.get("new_clusters"),
        "areas_kept": match.get("matched"),
        "areas_disappeared": len(match.get("disappeared", []) or []),
        "merges": match.get("merges", []),
        "splits": match.get("splits", []),
    })
    return summary


def publish(cfg: Config, meta: Dict, report: Dict, staging: Optional[str] = None) -> Optional[str]:
    gates = report.get("gates", {})
    if not gates.get("passed", False):
        msg = f"Validation gates failed: {gates.get('failed_required')}"
        if cfg["validation"]["enforce_gates"]:
            raise PublishError(msg + " — refusing to publish (validation.enforce_gates is true).")
        log.warning("%s — publishing anyway (gates are advisory in this config).", msg)
    if not cfg["release"]["enabled"]:
        return None
    staging = staging or cfg.processed_dir
    rid = _new_id(cfg)
    target = os.path.join(cfg.releases_dir, rid)
    previous = current_release(cfg)
    shutil.copytree(staging, target)
    save_json(build_manifest(cfg, rid, meta, report), os.path.join(target, "manifest.json"))
    save_json(build_changelog(cfg, target, previous, meta), os.path.join(target, "changelog.json"))
    for name in ("validation_report.json", "validation_report.md"):
        src = os.path.join(cfg.reports_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(target, name))
    set_current(cfg, rid)
    prune(cfg)
    log.info("Published release %s -> %s", rid, target)
    return rid


def set_current(cfg: Config, release_id: str) -> None:
    if release_id not in list_releases(cfg):
        raise FileNotFoundError(f"Release '{release_id}' not found")
    with open(os.path.join(cfg.releases_dir, CURRENT_FILE), "w", encoding="utf-8") as fh:
        fh.write(release_id)


def prune(cfg: Config) -> List[str]:
    releases = list_releases(cfg)
    keep = set(releases[-int(cfg["release"]["keep_last"]):])
    keep |= {r for r in (cfg["release"].get("pin"), current_release(cfg)) if r}
    removed = [r for r in releases if r not in keep]
    for r in removed:
        shutil.rmtree(os.path.join(cfg.releases_dir, r), ignore_errors=True)
        log.info("Pruned old release %s", r)
    return removed


def main() -> None:
    p = argparse.ArgumentParser(description="Manage ScholarGrid data releases")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    rb = sub.add_parser("rollback")
    rb.add_argument("release_id")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "list":
        cur = current_release(cfg)
        for r in list_releases(cfg):
            m = load_json(os.path.join(cfg.releases_dir, r, "manifest.json"))
            gates = (m.get("validation") or {}).get("gates", {}).get("passed")
            print(f"{'*' if r == cur else ' '} {r}  papers={m['corpus'].get('n_papers')}  gates_passed={gates}")
    else:
        set_current(cfg, args.release_id)
        print(f"CURRENT -> {args.release_id}")


if __name__ == "__main__":
    main()
