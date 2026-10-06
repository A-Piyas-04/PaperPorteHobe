"""Data quality checks, run on every build.

Each check returns pass/fail with the observed value and the threshold. When
``quality.fail_on_error`` is true (production) a failing check stops the build
before any expensive stage runs. Results go into the run metadata and the
release manifest.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .config import Config
from .text import title_key
from .utils import get_logger

log = get_logger("quality")


class DataQualityError(RuntimeError):
    pass


def _check(name: str, passed: bool, value, threshold, detail: str = "") -> Dict:
    return {"check": name, "passed": bool(passed), "value": value,
            "threshold": threshold, "detail": detail}


def span_months(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return round((df["date"].max() - df["date"].min()).days / 30.44, 1)


def run_checks(df: pd.DataFrame, cfg: Config, previous_count: Optional[int] = None) -> Dict:
    q = cfg["quality"]
    checks: List[Dict] = []

    span = span_months(df)
    checks.append(_check("time_span_months", span >= q["min_span_months"], span, q["min_span_months"]))

    if len(df):
        months = pd.period_range(df["date"].min(), df["date"].max(), freq="M").strftime("%Y-%m")
        per_month = df["year_month"].value_counts().reindex(months, fill_value=0)
        worst = int(per_month.min())
        gaps = [m for m, n in per_month.items() if n < q["min_papers_per_month"]]
        checks.append(_check("min_papers_per_month", not gaps, worst, q["min_papers_per_month"],
                             f"months below threshold: {gaps[:12]}" if gaps else ""))
        share = df["primary_category"].value_counts(normalize=True)
        top_cat, top_share = str(share.index[0]), round(float(share.iloc[0]), 4)
        checks.append(_check("max_category_share", top_share <= q["max_category_share"],
                             top_share, q["max_category_share"], f"largest: {top_cat}"))
    else:
        checks.append(_check("non_empty_corpus", False, 0, 1))

    missing = 0.0
    if len(df):
        empty = (df["title"].astype(str).str.strip() == "") | (df["abstract"].astype(str).str.strip() == "")
        missing = round(float(empty.mean()), 5)
    checks.append(_check("missing_title_abstract_rate", missing <= q["max_missing_rate"],
                         missing, q["max_missing_rate"]))

    dup_ids = int(df["arxiv_id"].duplicated().sum()) if len(df) else 0
    dup_titles = int(df["title"].map(title_key).duplicated().sum()) if len(df) else 0
    checks.append(_check("duplicates_after_dedup", dup_ids + dup_titles == 0, dup_ids + dup_titles, 0))

    if previous_count:
        change = round(abs(len(df) - previous_count) / previous_count, 4)
        checks.append(_check("row_count_change", change <= q["max_row_change"], change,
                             q["max_row_change"], f"previous={previous_count}, now={len(df)}"))

    failed = [c["check"] for c in checks if not c["passed"]]
    result = {"passed": not failed, "failed": failed, "checks": checks}
    for c in checks:
        (log.info if c["passed"] else log.warning)(
            "quality %-28s %s (value=%s, threshold=%s) %s", c["check"],
            "PASS" if c["passed"] else "FAIL", c["value"], c["threshold"], c["detail"])
    if failed and q["fail_on_error"]:
        raise DataQualityError(f"Data quality checks failed: {failed}")
    return result
