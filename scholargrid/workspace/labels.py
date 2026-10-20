"""Sub-topic labels: c-TF-IDF keywords (always), top facets, and LLM names + summaries (packed).

The query's own words are removed from the candidate keywords, so a "graph neural networks" workspace
doesn't label every sub-topic "Graph Neural Networks".
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..labeling import _c_tf_idf, _stem, pretty_label
from ..llm.prompts import wrap_untrusted

FACET_TYPES = ("task", "method", "dataset", "metric", "domain", "language")
_WORD = re.compile(r"[a-z0-9]+")


def _query_stems(query: str) -> set:
    return {_stem(w) for w in _WORD.findall(query.lower()) if len(w) > 1}


def keyword_labels(docs: Dict[int, str], query: str, top_k: int = 8) -> Dict[int, List[str]]:
    """{subtopic id: ranked keywords} from one aggregated document per sub-topic."""
    from sklearn.feature_extraction.text import CountVectorizer

    ids = [i for i, d in docs.items() if d.strip()]
    if not ids:
        return {i: [] for i in docs}
    try:
        vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        counts = vec.fit_transform([docs[i] for i in ids]).toarray().astype(float)
    except ValueError:                  # only stop words
        return {i: [] for i in docs}
    terms = np.array(vec.get_feature_names_out())
    banned = _query_stems(query)
    ok = np.array([not ({_stem(w) for w in t.split()} & banned) for t in terms])
    scores = _c_tf_idf(counts) * ok
    out: Dict[int, List[str]] = {i: [] for i in docs}
    for row, sid in enumerate(ids):
        order = np.argsort(-scores[row], kind="stable")[: top_k * 3]
        kws: List[str] = []
        for j in order:
            if scores[row, j] <= 0:
                break
            t = str(terms[j])
            if any(t in k or k in t for k in kws):      # skip "language" when "language model" is in
                continue
            kws.append(t)
            if len(kws) >= top_k:
                break
        out[sid] = kws
    return out


def keyword_label(keywords: Sequence[str]) -> str:
    return pretty_label(list(keywords)) if keywords else "Other papers"


def top_facets(facets: pd.DataFrame, arxiv_ids: Sequence[str], n: int = 5) -> Dict[str, List[List]]:
    """facet -> [[canonical value, paper count], ...] over the member papers (methods exclude baselines)."""
    if facets is None or not len(facets) or not len(arxiv_ids):
        return {}
    col = "value" if "value" in facets.columns else "value_raw"
    sub = facets[facets["arxiv_id"].astype(str).isin(set(arxiv_ids))]
    sub = sub[~((sub["facet"] == "method") & (sub["role"] == "baseline"))]
    out: Dict[str, List[List]] = {}
    for facet in FACET_TYPES:
        vals = sub.loc[sub["facet"] == facet].drop_duplicates(["arxiv_id", col])[col]
        top = Counter(vals.astype(str)).most_common(n)
        if top:
            out[facet] = [[v, int(c)] for v, c in top]
    return out


def evidence_lines(papers: Sequence[Any], profiles: Optional[Dict[str, str]], n: int) -> str:
    lines = []
    for p in papers[:n]:
        summary = (profiles or {}).get(p.paper_id) or (p.abstract.split(". ")[0] if p.abstract else "")
        lines.append(f"- {p.title} :: {summary[:240]}")
    return wrap_untrusted("evidence", "\n".join(lines))


def facets_text(top: Dict[str, List[List]]) -> str:
    parts = [f"{facet}: " + ", ".join(str(v) for v, _ in vals[:4]) for facet, vals in top.items()
             if facet in ("task", "method", "dataset")]
    return "; ".join(parts) or "none extracted"


def llm_labels(llm: Any, query: str, items: List[Tuple[int, Dict[str, str]]]) -> Tuple[Dict[int, Any], List[Any]]:
    """Packed SubtopicLabel calls. Returns ({sid: SubtopicLabel}, [errors])."""
    from ..llm import LLMError
    from .prompts import SubtopicLabel

    if not items:
        return {}, []
    req = [(str(sid), {"query": query, **v}) for sid, v in items]
    try:
        results = llm.structured_many("label", "subtopic_label", SubtopicLabel, req)
    except LLMError as exc:
        return {}, [exc]
    got: Dict[int, Any] = {}
    errors: List[Any] = []
    for k, res in results.items():
        if isinstance(res, LLMError) or res is None:
            errors.append(res)
        else:
            got[int(k)] = res.value
    return got, errors
