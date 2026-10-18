"""Graded relevance judgments for the search benchmark.

    python -m scholargrid.judgments export [--out reports/judgments_todo.csv] [--top 20] [--sample 15]
    python -m scholargrid.judgments import reports/judgments_todo.csv

``export`` runs the benchmark queries (all, or ``--sample N`` spread across
areas) against the current bundle and writes the top results with the
heuristic grade for reference; reviewers fill the ``grade`` column (0/1/2).
``import`` merges filled rows into ``search_judgments.json``.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from .config import load_config
from .utils import load_json, save_json


def sample_queries(queries: list, n: int | None) -> list:
    """``n`` queries taken round-robin across areas (all when ``n`` is None)."""
    if n is None or n >= len(queries):
        return list(queries)
    by_area: dict = {}
    for q in queries:
        by_area.setdefault(q.get("area", ""), []).append(q)
    picked, depth = [], 0
    while len(picked) < n:
        for group in by_area.values():
            if depth < len(group) and len(picked) < n:
                picked.append(group[depth])
        depth += 1
    return picked


def export(out: str, top: int, config: str | None, sample: int | None = None) -> None:
    from .artifacts import load_bundle
    from .search import search
    from .validation import _grades

    cfg = load_config(config)
    b = load_bundle(cfg)
    spec = load_json(cfg.abspath(cfg["validation"]["queries_file"]))
    queries = sample_queries(spec["queries"], sample)
    text = (b.df["title"].astype(str) + " " + b.df["abstract"].astype(str)).str.lower()
    rows = []
    for q in queries:
        grades = _grades(b.df, q, text)
        res = search(q["query"], b.embedder, b.embeddings, b.df, b.labels, b.clusters_meta,
                     top_k=top, bm25=b.bm25, index=b.index, cfg=cfg,
                     phrase_index=b.phrase, acronyms=b.acronyms)
        for r, i in zip(res["results"], res["result_indices"]):
            rows.append({"query": q["query"], "rank": r["rank"], "arxiv_id": r["arxiv_id"],
                         "title": r["title"], "abstract": str(b.df.iloc[i]["abstract"])[:400],
                         "heuristic_grade": int(grades[i]), "grade": ""})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows for {len(queries)} queries to {out}")


def import_(path: str, config: str | None) -> None:
    cfg = load_config(config)
    target = cfg.abspath(cfg["validation"]["judgments_file"] or "configs/search_judgments.json")
    data = load_json(target) if os.path.exists(target) else {"judgments": {}}
    sheet = pd.read_csv(path, dtype={"arxiv_id": str})
    sheet = sheet[sheet["grade"].astype(str).str.strip().isin(["0", "1", "2", "0.0", "1.0", "2.0"])]
    for query, g in sheet.groupby("query"):
        data["judgments"].setdefault(query, {}).update(
            {str(a): int(float(v)) for a, v in zip(g["arxiv_id"], g["grade"])})
    save_json(data, target)
    print(f"Imported {len(sheet)} judgments for {sheet['query'].nunique()} queries into {target}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default="reports/judgments_todo.csv")
    e.add_argument("--top", type=int, default=20)
    e.add_argument("--sample", type=int, default=None, help="only this many queries, spread across areas")
    i = sub.add_parser("import")
    i.add_argument("path")
    args = p.parse_args()
    if args.cmd == "export":
        export(args.out, args.top, args.config, args.sample)
    else:
        import_(args.path, args.config)


if __name__ == "__main__":
    main()
