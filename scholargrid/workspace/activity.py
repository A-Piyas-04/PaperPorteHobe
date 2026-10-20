"""Activity per sub-topic: monthly counts and the share of recent papers.

Retrieval is biased toward the query, so raw counts are never turned into "growing" claims here; the
rigorous growth statistics stay on the Trends page. The UI calls a sub-topic "recently active" only with
>= 15 papers and recent_share >= 0.5, and says it is based on the papers retrieved.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

RECENTLY_ACTIVE_MIN_SIZE = 15
RECENTLY_ACTIVE_MIN_SHARE = 0.5


def _ts(date: str) -> Optional[pd.Timestamp]:
    if not date or len(date) < 7:
        return None
    try:
        return pd.Timestamp(date)
    except (ValueError, TypeError):
        return None


def newest_date(dates: Sequence[str]) -> Optional[pd.Timestamp]:
    ts = [t for t in (_ts(d) for d in dates) if t is not None]
    return max(ts) if ts else None


def activity(dates: Sequence[str], newest: Optional[pd.Timestamp]) -> Tuple[Dict[str, int], float]:
    """(monthly counts for full dates, share of papers within 12 months of ``newest``)."""
    ts = [t for t in (_ts(d) for d in dates) if t is not None]
    monthly = dict(sorted(Counter(t.strftime("%Y-%m") for t in ts).items()))
    if not ts or newest is None:
        return monthly, 0.0
    cut = newest - pd.DateOffset(months=12)
    return monthly, round(sum(1 for t in ts if t >= cut) / len(ts), 4)


def recently_active(size: int, recent_share: float) -> bool:
    return size >= RECENTLY_ACTIVE_MIN_SIZE and recent_share >= RECENTLY_ACTIVE_MIN_SHARE
