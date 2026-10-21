"""Make gap finding work for the topics you actually use, today.

The gap engine needs ``problems.parquet`` and friends in the served bundle. The full
pipeline builds those for the whole scope, which costs weeks of free LLM quota. This
script does the same work for just the papers already in your saved topics, writes the
tables straight into the served release, and clears the stale empty gap caches.

    python scripts/warm_topics.py                 # every saved topic
    python scripts/warm_topics.py --limit 200     # try a slice first
    python scripts/warm_topics.py --no-fetch      # reuse already-fetched full text only

Full text is fetched from arXiv at one request every 3 seconds (their rate limit), so
expect roughly 3 seconds per not-yet-fetched paper. Both stores are incremental: stop
with Ctrl-C and re-run, nothing already done is repeated. Without an LLM configured the
heuristic extractor runs instead, which needs no quota and still yields real quotes from
Limitations / Future Work sections.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholargrid.config import load_config
from scholargrid.release import resolve_bundle_dir


def workspace_ids(cfg) -> set:
    """arxiv_ids of every paper in every saved topic."""
    ids = set()
    for path in glob.glob(os.path.join(cfg.workspace_dir, "*.json")):
        if path.endswith(".gaps.json"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        for p in doc.get("papers") or []:
            pid = str(p.get("paper_id") or "")
            if pid:
                ids.add(pid)
    return ids


def source_frame(cfg, bundle_dir: str, ids: set) -> pd.DataFrame:
    """The papers to process, taken from whichever stored frame actually holds the topic papers.

    ``stages/enriched.parquet`` can be left over from a synthetic dev run, so it is used only
    when it really overlaps the saved topics; otherwise the served bundle is authoritative.
    """
    candidates = [os.path.join(bundle_dir, "papers.parquet"),
                  os.path.join(cfg.stages_dir, "enriched.parquet")]
    best, best_hits = None, -1
    for path in candidates:
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df["arxiv_id"] = df["arxiv_id"].astype(str)
        hits = len(ids & set(df["arxiv_id"]))
        print(f"  {os.path.relpath(path)}: {len(df)} papers, {hits} of your topic papers")
        if hits > best_hits:
            best, best_hits = df, hits
    if best is None:
        raise FileNotFoundError("No papers.parquet or enriched.parquet found.")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="only the N most-cited matched papers")
    ap.add_argument("--all", action="store_true",
                    help="the whole corpus, most-cited first, not just saved topics")
    ap.add_argument("--categories", default="",
                    help="with --all: comma-separated primary categories (default: every category)")
    ap.add_argument("--no-fetch", action="store_true", help="skip arXiv fetching; extract what is stored")
    ap.add_argument("--keep-caches", action="store_true", help="do not clear saved topic gap caches")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Both stages re-filter through select_scope; neutralise it so the chosen papers survive.
    cfg.raw["scope"] = {**cfg.raw.get("scope", {}), "categories": "all", "min_date": None,
                        "max_papers": None, "priority": "most_cited"}

    bundle_dir = resolve_bundle_dir(cfg)
    print(f"Serving bundle: {bundle_dir}")

    if args.all:
        df = pd.read_parquet(os.path.join(bundle_dir, "papers.parquet"))
        df["arxiv_id"] = df["arxiv_id"].astype(str)
        sub = df
        if args.categories:
            want = {c.strip() for c in args.categories.split(",") if c.strip()}
            sub = sub[sub["primary_category"].astype(str).isin(want)]
            print(f"Whole corpus, categories {sorted(want)}: {len(sub)} of {len(df)} papers.")
        else:
            print(f"Whole corpus: {len(sub)} papers.")
    else:
        ids = workspace_ids(cfg)
        if not ids:
            print("No saved topics found in " + cfg.workspace_dir + ". Open a topic in the app first.")
            return 1
        print("Looking for your topic papers:")
        df = source_frame(cfg, bundle_dir, ids)
        sub = df[df["arxiv_id"].isin(ids)].copy()
    if "cited_by_count" in sub.columns:
        sub = sub.sort_values("cited_by_count", ascending=False, kind="stable")
    if args.limit:
        sub = sub.head(args.limit)
    if not args.all:
        print(f"Topic papers: {len(ids)} saved, {len(sub)} of them in the corpus and processable.")
    if not len(sub):
        print("Nothing to process.")
        return 1

    if not args.no_fetch:
        from scholargrid.fulltext import run_fulltext

        print(f"\nFetching full text (~{len(sub) * 3 / 60:.0f} min worst case; Ctrl-C is safe)…")
        stats = run_fulltext(sub, cfg, progress=lambda n, t: print(f"  {n}/{t}", end="\r", flush=True))
        print(f"\n  full text: {stats['ok']} ok, {stats['unavailable']} unavailable, "
              f"{stats['errors']} errors, {stats['fetched_now']} fetched now")

    from scholargrid.extract import run_extract
    from scholargrid.llm import get_llm

    llm = get_llm(cfg)
    mode = "LLM" if llm.available() else "heuristic (no LLM configured — no quota used)"
    print(f"\nExtracting open problems with the {mode} extractor…")
    stats = run_extract(sub, cfg, llm=llm)
    print(f"  extracted: {stats['llm']} by LLM, {stats['heuristic']} heuristic, "
          f"{stats['problems_verified']} problems verified, {stats['requests_used']} requests used")
    if stats["quota_stopped"]:
        print("  free quota ran out; re-run tomorrow to upgrade the heuristic rows.")

    from scholargrid.artifacts import load_bundle
    from scholargrid.extract.bundle import write_extraction_bundle

    print("\nWriting the extraction tables into the served bundle…")
    bundle = load_bundle(cfg, directory=bundle_dir)
    written = write_extraction_bundle(cfg, bundle.df, bundle.embedder, bundle_dir)
    print(f"  {written['papers_with_extraction']} papers with extraction, "
          f"{written['problems']} problems, facet values: {written['facet_values']}")

    if not args.keep_caches:
        gone = 0
        for path in glob.glob(os.path.join(cfg.workspace_dir, "*.gaps.json")):
            os.remove(path)
            gone += 1
        print(f"  cleared {gone} stale gap cache(s) so topics recompute on next visit")

    if not written["problems"]:
        print("\nNo problems extracted. Most likely no full text was retrievable for these papers;"
              "\nre-run without --no-fetch, or check data/fulltext/ status.")
        return 1
    print("\nDone. Restart the app and open a topic -> 'Find open problems for this topic'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
