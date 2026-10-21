"""End-to-end check of the two things an end user comes here for: open problems, then a plan.

Drives the same library calls the Streamlit buttons drive, with no browser, so a failure
points at a real defect rather than at a click. Live sources are off by default to keep runs
deterministic; pass --live to include the novelty web checks the app would do.

    python scripts/smoke_e2e.py              # every saved topic
    python scripts/smoke_e2e.py --topic SATD # one topic, matched on its query
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholargrid.config import load_config
from scholargrid.release import resolve_bundle_dir

OK, BAD = "  [ok]", "  [FAIL]"


def load_workspaces(cfg, needle: str):
    from scholargrid.workspace.model import TopicWorkspace

    out = []
    for path in sorted(glob.glob(os.path.join(cfg.workspace_dir, "*.json"))):
        if path.endswith(".gaps.json"):
            continue
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        ws = TopicWorkspace.from_json(doc)
        if not needle or needle.lower() in ws.query.lower():
            out.append(ws)
    return out


def check_bundle(bundle) -> list:
    """The data the gap engine needs; each missing table silently disables detectors."""
    problems = []
    for name in ("problems", "facets", "findings", "profiles"):
        t = getattr(bundle, name, None)
        n = 0 if t is None else len(t)
        print(f"  {name:10s} {n:6d} rows" + ("   <-- EMPTY" if not n else ""))
        if not n:
            problems.append(name)
    pe = getattr(bundle, "problem_embeddings", None)
    print(f"  {'problem_vectors':10s} {0 if pe is None else len(pe):6d} rows")
    if pe is None and getattr(bundle, "problems", None) is not None and len(bundle.problems):
        problems.append("problem_embeddings")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--topic", default="", help="only topics whose query contains this")
    ap.add_argument("--live", action="store_true", help="allow live novelty checks (slow, networked)")
    ap.add_argument("--plan", action="store_true", help="also generate a plan for the best card")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from scholargrid.artifacts import load_bundle
    from scholargrid.llm import get_llm

    bundle_dir = resolve_bundle_dir(cfg)
    print(f"Bundle: {bundle_dir}")
    bundle = load_bundle(cfg, directory=bundle_dir)
    llm = get_llm(cfg)
    print(f"LLM: provider={cfg['llm']['provider']} available={llm.available()}")

    print("\n[1] Extraction tables in the served bundle")
    missing = check_bundle(bundle)
    if "problems" in missing:
        print(f"{BAD} no extracted problems -> every detector returns zero. "
              "Run: python scripts/warm_topics.py")
        return 1
    print(f"{OK} extraction data present")

    workspaces = load_workspaces(cfg, args.topic)
    if not workspaces:
        print(f"{BAD} no saved topics matched.")
        return 1

    from scholargrid.gaps.workspace import gaps_for_workspace

    failures = 0
    for ws in workspaces:
        print(f"\n[2] Topic: {ws.query!r} ({len(ws.papers)} papers, "
              f"{len(ws.bundle_indices())} in collection)")
        covered = 0
        if len(bundle.problems):
            ids = set(bundle.problems["arxiv_id"].astype(str))
            covered = sum(1 for p in ws.papers if p.paper_id in ids)
        print(f"  papers with extracted problems: {covered}")
        try:
            data = gaps_for_workspace(ws, bundle, None, cfg, llm, use_cache=False)
        except Exception:
            print(f"{BAD} gaps_for_workspace raised:")
            traceback.print_exc()
            failures += 1
            continue
        stats, cards = data.get("stats") or {}, data.get("cards") or []
        print(f"  candidates={stats.get('candidates')} judged_out={stats.get('judged_out')} "
              f"already_addressed={stats.get('already_addressed')} cards={len(cards)}")
        if not cards:
            print(f"{BAD} no gap cards -> the app shows an empty state and planning stays unreachable.")
            failures += 1
            continue
        print(f"{OK} {len(cards)} cards; types: {stats.get('by_type')}")
        top = cards[0]
        print(f"  top card: {top.get('title','')[:80]!r}")
        print(f"    evidence={len(top.get('evidence') or [])} "
              f"score={(top.get('scores') or {}).get('total')}")
        if not (top.get("evidence") or []):
            print(f"{BAD} top card has no evidence.")
            failures += 1

        if args.plan:
            print(f"\n[3] Plan for the top card")
            from scholargrid.plan.generate import generate_plan

            try:
                doc = generate_plan(top, bundle, cfg, llm, bundle.embedder, weeks=12,
                                    background="", compute_access="laptop", goal="", workspace=ws)
            except Exception:
                print(f"{BAD} generate_plan raised:")
                traceback.print_exc()
                failures += 1
                continue
            pl = doc.plan
            parts = {"milestones": pl.milestones, "experiments": pl.experiments,
                     "sub_questions": pl.sub_questions, "baselines": pl.baselines,
                     "datasets": pl.datasets, "metrics": pl.metrics, "risks": pl.risks}
            print(f"  mode={doc.generated_by.get('mode')} refs={len(doc.references)} "
                  f"warnings={len(doc.warnings)}")
            print(f"  question: {pl.research_question[:100]!r}")
            for k, v in parts.items():
                print(f"    {k:14s} {len(v)}")
            empty = [k for k, v in parts.items() if not v]
            for w in doc.warnings[:3]:
                print(f"    warn: {w}")
            if not pl.research_question or not pl.milestones:
                print(f"{BAD} plan is unusable (no research question or no milestones).")
                failures += 1
            else:
                print(f"{OK} plan generated" + (f" (empty: {', '.join(empty)})" if empty else ", all sections populated"))
    print("\n" + ("ALL GREEN" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
