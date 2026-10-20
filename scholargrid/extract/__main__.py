"""CLI for the extraction store.

    python -m scholargrid.extract show 2401.01234         # payload + verified problems with quotes
    python -m scholargrid.extract sample --n 20 --out reports/extraction_sample.md   # manual QA (plan 09)
    python -m scholargrid.extract stats                    # modes, verified rate, pending LLM, top facets
    python -m scholargrid.extract rerun --ids 2401.01234 --mode llm   # force re-extraction (debug)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import List, Optional

import pandas as pd

from ..config import Config, load_config
from ..utils import ensure_dir
from .store import ExtractionStore


def _titles(cfg: Config) -> pd.DataFrame:
    for path in (os.path.join(cfg.stages_dir, "enriched.parquet"), os.path.join(cfg.stages_dir, "corpus.parquet")):
        if os.path.exists(path):
            return pd.read_parquet(path, columns=["arxiv_id", "title"]).assign(
                arxiv_id=lambda d: d["arxiv_id"].astype(str))
    return pd.DataFrame(columns=["arxiv_id", "title"])


def _show(cfg: Config, arxiv_id: str) -> None:
    r = ExtractionStore(cfg).row(arxiv_id)
    if r is None:
        print(f"{arxiv_id}: no extraction stored")
        return
    print(f"{arxiv_id}: mode={r['mode']} model={r['model']} prompt v{r['prompt_version']} "
          f"problems {r['n_problems_verified']}/{r['n_problems_raw']} verified")
    print(json.dumps(json.loads(r["payload"]), indent=2, ensure_ascii=False))
    for p in json.loads(r["problems"]):
        print(f"\n[{p['section']}/{p['section_kind']}] ({p['problem_type']}, specificity {p['specificity']})")
        print(f"  {p['statement']}")
        print(f"  quote: \"{p['quote']}\"")


def _stats(cfg: Config) -> None:
    df = ExtractionStore(cfg).df
    print(f"store: {cfg.extract_dir}")
    print(f"papers: {len(df)}")
    if not len(df):
        return
    for mode, n in df["mode"].value_counts().items():
        print(f"  {mode:22s} {n:6d}")
    raw, ver = int(df["n_problems_raw"].sum()), int(df["n_problems_verified"].sum())
    print(f"problems: {ver} verified of {raw} proposed ({ver / raw:.0%} kept)" if raw else "problems: 0")
    print(f"pending an LLM pass (heuristic rows): {int(df['mode'].astype(str).str.startswith('heuristic').sum())}")
    print(f"papers at max LLM attempts: {int((df['llm_attempts'] >= int(cfg['extract']['max_llm_attempts'])).sum())}")
    counts: dict = {"tasks": Counter(), "datasets": Counter()}
    methods: Counter = Counter()
    for payload in df["payload"]:
        p = json.loads(payload)
        counts["tasks"].update(p.get("tasks", []))
        counts["datasets"].update(p.get("datasets", []))
        methods.update(m["name"] for m in p.get("methods", []))
    for name, c in (("tasks", counts["tasks"]), ("methods", methods), ("datasets", counts["datasets"])):
        print(f"top {name}: " + ", ".join(f"{k} ({v})" for k, v in c.most_common(10)))


def _sample(cfg: Config, n: int, out: str) -> None:
    df = ExtractionStore(cfg).df
    titles = _titles(cfg).set_index("arxiv_id")["title"].to_dict()
    sample = df.sample(min(n, len(df)), random_state=int(cfg["seed"])) if len(df) else df
    lines = ["# Extraction sample for manual QA", "",
             "For each paper count: (a) problems that are real author-stated problems, (b) quotes that are "
             "exact, (c) obvious missed problems.", ""]
    for r in sample.itertuples():
        p = json.loads(r.payload)
        aid = str(r.arxiv_id)
        lines += [f"## {titles.get(aid, aid)}", f"[{aid}](https://arxiv.org/abs/{aid}) · mode `{r.mode}` · "
                  f"model `{r.model}`", "", f"**Summary:** {p.get('one_line_summary', '')}", "",
                  f"- tasks: {', '.join(p.get('tasks', [])) or '-'}",
                  f"- methods: {', '.join(m['name'] + ' (' + m['role'] + ')' for m in p.get('methods', [])) or '-'}",
                  f"- datasets: {', '.join(p.get('datasets', [])) or '-'}",
                  f"- metrics: {', '.join(p.get('metrics', [])) or '-'}", "", "**Open problems (verified):**", ""]
        probs = json.loads(r.problems)
        if not probs:
            lines.append("- none")
        for q in probs:
            lines += [f"- {q['statement']} *({q['problem_type']}, {q['section_kind'] or q['section']})*",
                      f"  > {q['quote']}"]
        lines.append("")
    ensure_dir(os.path.dirname(os.path.abspath(out)))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(sample)} papers to {out}")


def _rerun(cfg: Config, ids: List[str], mode: str) -> None:
    from .run import run_extract

    path = os.path.join(cfg.stages_dir, "enriched.parquet")
    if not os.path.exists(path):
        raise SystemExit("run the pipeline up to `enrich` first")
    df = pd.read_parquet(path)
    df = df[df["arxiv_id"].astype(str).isin(set(ids))]
    ExtractionStore(cfg).drop(ids)
    cfg.raw["extract"]["mode"] = mode
    cfg.raw["scope"]["max_papers"] = None
    print(run_extract(df, cfg))


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m scholargrid.extract", description="Extraction store tools")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("show")
    sh.add_argument("arxiv_id")
    sh.add_argument("--config", default=argparse.SUPPRESS)
    ss = sub.add_parser("stats")
    ss.add_argument("--config", default=argparse.SUPPRESS)
    sm = sub.add_parser("sample")
    sm.add_argument("--n", type=int, default=20)
    sm.add_argument("--out", default="reports/extraction_sample.md")
    sm.add_argument("--config", default=argparse.SUPPRESS)
    sr = sub.add_parser("rerun")
    sr.add_argument("--ids", nargs="+", required=True)
    sr.add_argument("--mode", choices=["auto", "llm", "heuristic"], default="llm")
    sr.add_argument("--config", default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "show":
        _show(cfg, args.arxiv_id)
    elif args.cmd == "stats":
        _stats(cfg)
    elif args.cmd == "sample":
        _sample(cfg, args.n, args.out)
    elif args.cmd == "rerun":
        _rerun(cfg, args.ids, args.mode)


if __name__ == "__main__":
    main()
