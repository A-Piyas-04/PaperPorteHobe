"""Quality metrics from rater answers, plus the round report against the launch bar.

Majority voting across raters per item; small samples get Wilson intervals; agreement is the mean
pairwise Cohen's kappa over overlapping items. Empty inputs return None, never crash.
"""
from __future__ import annotations

import json
import os
import time
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import ensure_dir, save_json
from .sampling import wilson

LAUNCH_BAR = {"gap_precision": (">=", 0.60), "novelty_false_positive_rate": ("<=", 0.15),
              "problem_precision": (">=", 0.80), "facet_precision": (">=", 0.85),
              "plan_realistic": (">=", 3.5), "plan_grounded": (">=", 3.5), "plan_useful": (">=", 3.5)}


def _flatten(ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings is None or not len(ratings):
        return pd.DataFrame()
    crit = pd.DataFrame(list(ratings["criteria"].apply(lambda c: c if isinstance(c, dict) else {})))
    return pd.concat([ratings[["item_id", "rater"]].reset_index(drop=True), crit.reset_index(drop=True)], axis=1)


def _rate(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    return round(float(s.mean()), 3) if len(s) else None


def pairwise_kappa(df: pd.DataFrame, field: str) -> Optional[float]:
    """Mean Cohen's kappa over rater pairs on their overlapping items."""
    from sklearn.metrics import cohen_kappa_score

    if df is None or not len(df) or field not in df.columns:
        return None
    wide = df.pivot_table(index="item_id", columns="rater", values=field, aggfunc="first")
    vals = []
    for a, b in combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        if len(both) < 2:
            continue
        x, y = both[a].astype(str), both[b].astype(str)
        if (x == y).all():
            vals.append(1.0)
            continue
        k = cohen_kappa_score(x, y)
        if not np.isnan(k):
            vals.append(float(k))
    return round(float(np.mean(vals)), 3) if vals else None


def gap_metrics(ratings: pd.DataFrame, round_items: Dict[str, Dict]) -> Dict:
    """ratings: item_id, rater, criteria(dict). round_items: item_id -> card JSON (for type/confidence/novelty)."""
    df = _flatten(ratings)
    if not len(df) or "real" not in df.columns:
        return {"n_items": 0, "gap_precision": None, "novelty_false_positive_rate": None}
    df["good"] = (df["real"] == "yes") & (df.get("open") != "no")
    per_item = df.groupby("item_id")["good"].mean() >= 0.5          # majority across raters
    meta = pd.DataFrame.from_dict({k: {"type": v.get("type"), "confidence": v.get("confidence"),
                                       "novelty": (v.get("novelty") or {}).get("status")}
                                   for k, v in round_items.items()}, orient="index")
    k = int(per_item.sum())
    out: Dict[str, Any] = {"n_items": int(per_item.size), "n_ratings": int(len(df)),
                           "gap_precision": round(float(per_item.mean()), 3), "gap_precision_ci": wilson(k, int(per_item.size))}
    novel = meta.index[meta["novelty"] == "novel"] if len(meta) else []
    open_no = df.groupby("item_id")["open"].agg(lambda s: (s == "no").mean() >= 0.5) if "open" in df else pd.Series()
    fp = open_no.reindex(novel).dropna()
    out["novelty_false_positive_rate"] = round(float(fp.mean()), 3) if len(fp) else None
    out["n_novel"] = int(len(fp))
    for key in ("type", "confidence"):
        out[f"precision_by_{key}"] = {str(kk): {"precision": _rate(per_item.reindex(g.index)), "n": int(per_item.reindex(g.index).notna().sum())}
                                      for kk, g in meta.groupby(key)} if len(meta) else {}
    out["feasible_yes_rate"] = _rate(df["feasible"] == "yes") if "feasible" in df else None
    out["evidence_supports_rate"] = _rate(df["supported"] == "yes") if "supported" in df else None
    out["kappa_real"] = pairwise_kappa(df, "real")
    out["kappa_open"] = pairwise_kappa(df, "open")
    return out


def extraction_metrics(ratings: pd.DataFrame) -> Dict:
    df = _flatten(ratings)
    out: Dict[str, Any] = {"problem_precision": None, "faithful_rate": None, "facet_precision": None,
                           "facet_recall": None}
    if not len(df):
        return out
    prob = df[df["item_id"].astype(str).str.startswith("problem:")]
    if len(prob) and "stated" in prob:
        per = prob.groupby("item_id")["stated"].agg(lambda s: (s == "yes").mean() >= 0.5)
        out.update(problem_precision=round(float(per.mean()), 3), problem_precision_ci=wilson(int(per.sum()), int(per.size)),
                   n_problems=int(per.size))
        out["faithful_rate"] = _rate(prob["faithful"] == "yes") if "faithful" in prob else None
        out["specificity_mean"] = _rate(pd.to_numeric(prob.get("specificity"), errors="coerce")) if "specificity" in prob else None
    fac = df[df["item_id"].astype(str).str.startswith("facets:")]
    if len(fac) and "n_shown" in fac:
        shown = pd.to_numeric(fac["n_shown"], errors="coerce").fillna(0)
        wrong = pd.to_numeric(fac["n_wrong"], errors="coerce").fillna(0)
        missing = pd.to_numeric(fac.get("n_missing", 0), errors="coerce").fillna(0)
        correct = (shown - wrong).clip(lower=0)
        out["facet_precision"] = round(float(correct.sum() / shown.sum()), 3) if shown.sum() else None
        out["facet_recall"] = round(float(correct.sum() / (correct.sum() + missing.sum())), 3) \
            if (correct.sum() + missing.sum()) else None
        out["n_facet_items"] = int(len(fac))
    out["kappa_stated"] = pairwise_kappa(prob, "stated") if len(prob) else None
    return out


def search_metrics(ratings: pd.DataFrame, k: int = 10) -> Dict:
    """nDCG@k from human grades (majority -> mean grade per (query, paper)), in rank order."""
    from ..validation import _dcg

    df = _flatten(ratings)
    if not len(df) or "grade" not in df:
        return {"ndcg_at_10": None, "n_queries": 0}
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    df["query"] = df["item_id"].astype(str).str.slice(len("search:")).str.rsplit("|", n=1).str[0]
    df["rank"] = pd.to_numeric(df.get("rank"), errors="coerce")
    scores = []
    for _, g in df.groupby("query"):
        per = g.groupby("item_id").agg(grade=("grade", "mean"), rank=("rank", "min")).sort_values("rank")
        gains = per["grade"].fillna(0).tolist()[:k]
        ideal = sorted(per["grade"].fillna(0).tolist(), reverse=True)[:k]
        idcg = _dcg(ideal)
        if idcg > 0:
            scores.append(_dcg(gains) / idcg)
    return {"ndcg_at_10": round(float(np.mean(scores)), 3) if scores else None, "n_queries": len(scores)}


def plan_metrics(ratings: pd.DataFrame) -> Dict:
    df = _flatten(ratings)
    out: Dict[str, Any] = {}
    for f in ("realistic", "grounded", "useful"):
        if len(df) and f in df:
            v = pd.to_numeric(df[f], errors="coerce").dropna()
            out[f"plan_{f}"] = round(float(v.mean()), 2) if len(v) else None
            out[f"plan_{f}_sd"] = round(float(v.std(ddof=0)), 2) if len(v) else None
        else:
            out[f"plan_{f}"] = None
    out["n_plans"] = int(df["item_id"].nunique()) if len(df) else 0
    return out


def reading_metrics(ratings: pd.DataFrame) -> Dict:
    df = _flatten(ratings)
    out: Dict[str, Any] = {"n_paths": int(df["item_id"].nunique()) if len(df) else 0}
    for f in ("right_papers", "order"):
        v = pd.to_numeric(df[f], errors="coerce").dropna() if len(df) and f in df else pd.Series(dtype=float)
        out[f"reading_{f}"] = round(float(v.mean()), 2) if len(v) else None
    return out


def _bar(name: str, value: Optional[float]) -> str:
    op, thr = LAUNCH_BAR[name]
    if value is None:
        return f"- Launch bar · {name}: not measured ({op} {thr})"
    ok = value >= thr if op == ">=" else value <= thr
    return f"- Launch bar · {name}: **{value}** {op} {thr} {'✅' if ok else '❌'}"


def _worst(ratings: pd.DataFrame, items: Dict[str, Dict], kind: str, n: int = 10) -> List[Dict]:
    df = _flatten(ratings)
    if not len(df):
        return []
    if kind == "gaps" and "real" in df:
        bad = df.assign(bad=(df["real"] != "yes").astype(int) + (df.get("open") == "no").astype(int))
    elif kind == "extraction" and "stated" in df:
        bad = df.assign(bad=(df["stated"] != "yes").astype(int))
    else:
        return []
    bad["comment"] = ratings["comment"].fillna("").astype(str).tolist() if "comment" in ratings else ""
    worst = bad.groupby("item_id").agg(bad=("bad", "mean"), comment=("comment", lambda s: " | ".join(x for x in s if x)))
    out = []
    for iid, r in worst.sort_values("bad", ascending=False).head(n).iterrows():
        payload = (items.get(str(iid)) or {})
        out.append({"item_id": str(iid), "title": payload.get("title") or payload.get("statement", "")[:80],
                    "badness": round(float(r["bad"]), 2), "comments": r["comment"]})
    return out


def report(cfg: Config, rnd: Dict, ratings: List[Dict]) -> Dict:
    """Compute the round's metrics, write reports/eval/<batch>.{md,json}, append history.jsonl."""
    kind = rnd["kind"]
    items = {it["item_id"]: it["payload"] for it in rnd["items"]}
    df = pd.DataFrame(ratings)
    if kind == "gaps":
        m = gap_metrics(df, items)
        bars = ["gap_precision", "novelty_false_positive_rate"]
    elif kind == "extraction":
        m = extraction_metrics(df)
        bars = ["problem_precision", "facet_precision"]
    elif kind == "search":
        m = search_metrics(df)
        bars = []
    elif kind == "plans":
        m = plan_metrics(df)
        bars = ["plan_realistic", "plan_grounded", "plan_useful"]
    else:
        m = reading_metrics(df)
        bars = []
    raters = sorted({r["rater"] for r in ratings})
    m.update(batch=rnd["name"], kind=kind, n_raters=len(raters), n_ratings=len(ratings), at=time.time())
    worst = _worst(df, items, kind)
    lines = [f"# Evaluation report: {rnd['name']} ({kind})", "",
             f"{len(ratings)} ratings by {len(raters)} raters on {len(rnd['items'])} items.", "", "## Headline", ""]
    lines += [_bar(b, m.get(b)) for b in bars] or ["- (no launch bar for this kind)"]
    if kind == "gaps":
        ci = m.get("gap_precision_ci")
        lines += ["", f"Gap precision 95% interval: {ci}", f"Agreement: kappa(real) = {m.get('kappa_real')}, "
                  f"kappa(open) = {m.get('kappa_open')}", "", "## Precision by type", "", "| Type | Precision | n |", "|---|---|---|"]
        lines += [f"| {k} | {v['precision']} | {v['n']} |" for k, v in m.get("precision_by_type", {}).items()]
        lines += ["", "## Calibration (precision by confidence)", "", "| Confidence | Precision | n |", "|---|---|---|"]
        lines += [f"| {k} | {v['precision']} | {v['n']} |" for k, v in m.get("precision_by_confidence", {}).items()]
    else:
        lines += ["", "## Metrics", ""] + [f"- {k}: {v}" for k, v in m.items() if k not in ("at",)]
    if worst:
        lines += ["", "## Worst-rated items", "", "| Item | Title | Badness | Comments |", "|---|---|---|---|"]
        lines += [f"| {w['item_id']} | {str(w['title'])[:70]} | {w['badness']} | {str(w['comments'])[:200]} |" for w in worst]
    out_dir = ensure_dir(cfg.abspath(cfg["evaluation"]["report_dir"]))
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in rnd["name"])
    with open(os.path.join(out_dir, f"{safe}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    save_json({"metrics": m, "worst": worst}, os.path.join(out_dir, f"{safe}.report.json"))
    with open(os.path.join(out_dir, "history.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, default=str) + "\n")
    return m


def latest_history(cfg: Config, kind: str = "gaps") -> Optional[Dict]:
    path = os.path.join(cfg.abspath(cfg["evaluation"]["report_dir"]), "history.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("kind") == kind:
                last = d
    return last
