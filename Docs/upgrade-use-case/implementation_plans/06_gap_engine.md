# Plan 06: The Gap Engine (detectors, verification, novelty check, scoring, Gap Cards)

> **Prerequisites:** plans 01–05 are merged. The bundle has `problems` (verified quotes + embeddings), `facets`, `findings`, `profiles`, `foundations`, and `reading_paths`; the topic workspace exists.
> **Produces:**
> 1. `scholargrid/gaps/` with **six detectors** that produce evidence-backed gap *candidates*,
> 2. an LLM **judge** for plausibility and conflicts, and a **synthesizer** that writes plain-language Gap Cards grounded only in evidence,
> 3. **text verification** (no invented entities) and a **novelty check** against recent collection papers and live sources,
> 4. transparent **scoring and confidence**,
> 5. an offline stage **`gaps`** (per-area cards in `gaps.json`), an online **workspace mode**, and the **Open problems** page.
>
> **Why:** this is the product's reason to exist. Plan 03 made sure every quote is real; this plan makes sure every gap is (a) supported by several papers, (b) explained honestly, and (c) checked against newer work.

---

## 1. Prompt to paste into Cursor

```text
Implement plan file 06_gap_engine.md. It is the largest plan; do it in the order of §4-§15 and run
ruff/mypy/pytest after each section. Follow 00_INDEX_READ_FIRST.md: only scholargrid.llm talks to
LLM providers (free tiers; mind the request quota); every user-visible claim must trace to Evidence; never show a card without
"why_it_might_not_be_a_gap" and a novelty status. Detectors must be pure functions of a GapContext
(no I/O) so they are unit-testable with hand-built tables. Finish with §16.
```

---

## 2. Pipeline of one scope (an area offline, or a workspace online)

```
GapContext(scope papers, facets, problems+vectors, findings, profiles, refs)
      │
      ├─ detect_stated          recurring author-stated problems (+ long-standing flag)
      ├─ detect_combinatorial   method×task, method×domain, task×language, method×modality holes
      ├─ detect_evaluation      methods evaluated narrowly (few benchmarks, no real-world/robustness, English-only)
      ├─ detect_unsettled       conflicting reported results on the same task/dataset/metric
      └─ detect_bridge          semantically close sub-areas that rarely cite each other
      ▼
 candidates (with Evidence, stats, seed statement)
      ▼  pre-rank by evidence strength, keep top N
 judge (LLM; combinatorial plausibility, unsettled conflicts)  → drop implausible
      ▼
 synthesize (LLM; GapDraft from evidence only) → verify text (entity guard) → retry once → template fallback
      ▼
 novelty check (collection recent papers; + live sources online) → judge "already addressed?"
      ▼
 score + confidence → drop "already_addressed" → top K GapCards
```

---

## 3. Config (`scholargrid/schema.py` + `configs/config.yaml`)

```python
GapType = Literal["stated_open_problem", "long_standing_problem", "unexplored_combination", "evaluation_gap",
                  "unsettled_question", "missing_bridge"]


class Gaps(_Section):
    enabled: bool = True
    detectors: List[Literal["stated", "combinatorial", "evaluation", "unsettled", "bridge"]] = Field(
        default_factory=lambda: ["stated", "combinatorial", "evaluation", "unsettled", "bridge"])
    # stated / long-standing
    problem_cluster_distance: float = Field(default=0.25, gt=0, lt=1)   # cosine distance (average linkage)
    min_support: int = Field(default=3, ge=2)
    addressed_sim: float = Field(default=0.70, gt=0, le=1)
    longstanding_min_months: int = Field(default=24, ge=6)
    # combinatorial
    combo_pairs: List[str] = Field(default_factory=lambda: ["method:task", "method:domain", "task:language",
                                                            "method:modality"])
    min_marginal: int = Field(default=5, ge=2)
    min_expected: float = Field(default=3.0, gt=0)
    max_observed: int = Field(default=0, ge=0)
    max_p: float = Field(default=0.05, gt=0, le=1)
    min_bridge_sim: float = Field(default=0.60, gt=0, le=1)
    require_bridge: bool = True
    combo_top_values: int = Field(default=60, ge=5)
    # evaluation
    eval_min_papers: int = Field(default=5, ge=2)
    eval_max_datasets: int = Field(default=2, ge=1)
    eval_min_known: float = Field(default=0.5, gt=0, le=1)
    # unsettled
    unsettled_min_papers: int = Field(default=2, ge=2)
    # bridge
    bridge_sim_quantile: float = Field(default=0.9, gt=0, lt=1)
    bridge_max_ratio: float = Field(default=0.25, gt=0, le=1)
    bridge_min_ref_coverage: float = Field(default=0.3, gt=0, le=1)
    # finalisation
    max_candidates_per_scope: int = Field(default=30, ge=1)
    max_cards_per_scope: int = Field(default=12, ge=1)
    judge_with_llm: bool = True
    synthesize_with_llm: bool = True
    novelty_check: bool = True
    novelty_recent_months: int = Field(default=18, ge=1)
    novelty_collection_top: int = Field(default=10, ge=1)
    novelty_top_judge: int = Field(default=4, ge=0)
    novelty_live_queries: int = Field(default=3, ge=0)
    novelty_ttl_days: int = Field(default=30, ge=1)
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "evidence": 0.30, "novelty": 0.25, "feasibility": 0.20, "specificity": 0.15, "recency": 0.10})
    online_max_new_cards: int = Field(default=8, ge=1)
    keep_already_addressed: bool = False     # show them in an "Already explored" list instead of dropping
```

Add `gaps: Gaps = Field(default_factory=Gaps)` to `ConfigModel`, and the YAML block with comments explaining every threshold in one line each.

---

## 4. Models: `scholargrid/gaps/model.py`

```python
"""Gap candidates (internal) and Gap Cards (shown to users, stored in bundles/caches/projects)."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

EvidenceRole = Literal["states_problem", "related_attempt", "method_use", "task_use", "bridge", "narrow_eval",
                       "conflicting_result", "side_a", "side_b"]
NoveltyStatus = Literal["not_checked", "novel", "partially_explored", "already_addressed"]
Level = Literal["low", "medium", "high"]

TYPE_LABELS = {
    "stated_open_problem": "Open problem stated by authors",
    "long_standing_problem": "Long-standing open problem",
    "unexplored_combination": "Unexplored combination",
    "evaluation_gap": "Evaluation gap",
    "unsettled_question": "Unsettled question",
    "missing_bridge": "Missing connection between areas",
}
TYPE_HELP = {
    "stated_open_problem": "Several papers' authors say this is still unsolved.",
    "long_standing_problem": "Authors have called this open for two years or more, and still do.",
    "unexplored_combination": "A method and a problem that fit together, but no paper here combines them.",
    "evaluation_gap": "A popular method that has only been tested in narrow conditions.",
    "unsettled_question": "Papers report conflicting results on the same question.",
    "missing_bridge": "Two closely related lines of work that rarely cite each other.",
}


class Evidence(BaseModel):
    eid: str                         # "E1", "E2", ... unique within a card
    paper_id: str
    title: str
    year: Optional[int] = None
    url: str = ""
    role: EvidenceRole
    quote: str = ""                  # ONLY verified quotes from problems.parquet; else ""
    section: str = ""                # e.g. "Limitations"
    anchor_url: str = ""             # https://arxiv.org/html/<id>#<anchor> when known
    note: str = ""                   # e.g. "evaluated on: SQuAD, NQ" or a paraphrased finding (labelled)


class NoveltyHit(BaseModel):
    paper_id: str
    title: str
    url: str
    date: str = ""
    verdict: Literal["addresses", "partially", "unrelated", "unjudged"]
    reason: str = ""
    quote: str = ""                  # verified against the abstract, else ""


class NoveltyCheck(BaseModel):
    status: NoveltyStatus = "not_checked"
    scope: Literal["none", "collection", "collection+live"] = "none"
    checked_at: Optional[float] = None
    queries: List[str] = Field(default_factory=list)
    n_candidates: int = 0
    hits: List[NoveltyHit] = Field(default_factory=list)
    method: Literal["llm", "heuristic", "none"] = "none"


class Scores(BaseModel):
    evidence: float = 0.0
    novelty: float = 0.0
    feasibility: float = 0.0
    specificity: float = 0.0
    recency: float = 0.0
    total: float = 0.0
    explain: Dict[str, str] = Field(default_factory=dict)


class GapCard(BaseModel):
    id: str
    type: str
    title: str
    statement: str
    why_it_matters: str = ""
    why_it_might_not_be_a_gap: List[str]
    first_steps: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    data_needs: str = ""
    compute: str = "unknown"
    feasibility: Level = "medium"
    beginner_friendly: bool = False
    evidence: List[Evidence]
    novelty: NoveltyCheck = Field(default_factory=NoveltyCheck)
    scores: Scores = Field(default_factory=Scores)
    confidence: Level = "low"
    key_terms: Dict[str, str] = Field(default_factory=dict)
    stats: Dict = Field(default_factory=dict)
    scope: Dict = Field(default_factory=dict)          # {"kind": "area"|"workspace", "id": ..., "label": ...}
    generated_by: Dict = Field(default_factory=dict)   # {"model", "prompt_version", "synthesized": bool}
    created_at: float = Field(default_factory=time.time)
    version: int = 1


@dataclass
class GapCandidate:
    type: str
    scope: Dict
    key_terms: Dict[str, str]
    seed_statement: str
    evidence: List[Evidence]
    stats: Dict = field(default_factory=dict)
    specificity: float = 0.5           # 0..1
    strength: float = 0.0              # detector-specific pre-rank score (0..1)
    problem_ids: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        key = f"{self.type}|{sorted(self.key_terms.items())}|{self.scope.get('id')}|{self.seed_statement[:80]}"
        return "g" + hashlib.sha1(key.encode()).hexdigest()[:12]
```

---

## 5. Context: `scholargrid/gaps/context.py`

```python
"""Everything a detector may read for one scope. Detectors never touch disk or network."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import l2_normalize


@dataclass
class GapContext:
    cfg: object
    scope: Dict                         # {"kind", "id", "label"}
    papers: pd.DataFrame                # bundle rows in scope (keeps original bundle index in column "bidx")
    vectors: np.ndarray                 # unit vectors aligned with papers
    facets: pd.DataFrame                # arxiv_id, facet, value, role (canonical)
    problems: pd.DataFrame              # problems in scope (+ "vrow": row into problem_vectors)
    problem_vectors: Optional[np.ndarray]
    findings: pd.DataFrame
    profiles: pd.DataFrame
    ref_date: pd.Timestamp
    all_papers: pd.DataFrame = field(default_factory=pd.DataFrame)   # whole bundle df (bridge/novelty)
    all_vectors: Optional[np.ndarray] = None
    value_vectors: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)  # facet -> value -> unit vec
    sub_labels: Optional[np.ndarray] = None   # sub-group labels within scope (workspace sub-topics) for bridges

    def paper_row(self, arxiv_id: str) -> Optional[pd.Series]:
        hit = self.papers[self.papers["arxiv_id"] == arxiv_id]
        return hit.iloc[0] if len(hit) else None


def build_context(bundle, idx: np.ndarray, scope: Dict, cfg, embedder=None,
                  sub_labels: Optional[np.ndarray] = None) -> GapContext:
    papers = bundle.df.iloc[idx].copy()
    papers["bidx"] = idx
    ids = set(papers["arxiv_id"].astype(str))
    sel = lambda df: df[df["arxiv_id"].isin(ids)].reset_index(drop=True) if len(df) else df  # noqa: E731
    probs = bundle.problems.copy() if len(bundle.problems) else pd.DataFrame()
    if len(probs):
        probs["vrow"] = np.arange(len(probs))
        probs = probs[probs["arxiv_id"].isin(ids)].reset_index(drop=True)
    pvec = bundle.problem_embeddings
    if pvec is None and len(probs) and embedder is not None:
        pvec_rows = l2_normalize(np.asarray(embedder.transform(probs["statement"].tolist()), dtype=np.float32))
        probs["vrow"] = np.arange(len(probs))
        pvec = pvec_rows
    facets = sel(bundle.facets)
    ctx = GapContext(cfg, scope, papers.reset_index(drop=True),
                     l2_normalize(np.asarray(bundle.embeddings[idx], dtype=np.float32)), facets, probs,
                     None if pvec is None else l2_normalize(np.asarray(pvec, dtype=np.float32)),
                     sel(bundle.findings), sel(bundle.profiles), pd.Timestamp(bundle.df["date"].max()),
                     bundle.df, bundle.embeddings, sub_labels=sub_labels)
    if embedder is not None and len(facets):
        for facet in ("task", "method", "domain", "dataset"):
            vals = facets.loc[facets["facet"] == facet, "value"].value_counts().head(200).index.tolist()
            if vals:
                vecs = l2_normalize(np.asarray(embedder.transform(vals), dtype=np.float32))
                ctx.value_vectors[facet] = dict(zip(vals, vecs))
    return ctx
```

**Evidence helper** (`gaps/evidence.py`). Every detector creates evidence through it, so URLs and anchors are consistent:

```python
def evidence_for(ctx, arxiv_id: str, role: str, *, quote: str = "", section: str = "", anchor: str = "",
                 note: str = "") -> Evidence:
    r = ctx.paper_row(arxiv_id)
    if r is None:
        r = ctx.all_papers[ctx.all_papers["arxiv_id"] == arxiv_id].iloc[0]
    anchor_url = f"https://arxiv.org/html/{arxiv_id}#{anchor}" if anchor else ""
    return Evidence(eid="", paper_id=arxiv_id, title=str(r["title"]), year=int(pd.Timestamp(r["date"]).year),
                    url=f"https://arxiv.org/abs/{arxiv_id}", role=role, quote=quote, section=section,
                    anchor_url=anchor_url, note=note)


def number_evidence(items: list[Evidence], limit: int = 8) -> list[Evidence]:
    """Dedupe by (paper_id, role), cap, assign E1..En."""
```

---

## 6. Detector: stated open problems (+ long-standing): `gaps/detectors/stated.py`

```python
"""Recurring author-stated open problems. A cluster of similar problem statements from >= min_support
distinct papers is a candidate. Later papers that look like they address it become 'related_attempt'
evidence (the novelty step makes the final call)."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..evidence import evidence_for, number_evidence
from ..model import GapCandidate

_SECTION_NAMES = {"limitations": "Limitations", "future_work": "Future work", "discussion": "Discussion",
                  "conclusion": "Conclusion", "threats_to_validity": "Threats to validity",
                  "broader_impact": "Broader impact", "abstract": "Abstract", "introduction": "Introduction"}


def _cluster(x: np.ndarray, distance: float, seed: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering, KMeans

    if len(x) < 2:
        return np.zeros(len(x), dtype=int)
    if len(x) <= 4000:
        return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                       distance_threshold=distance).fit_predict(x)
    # Large scopes: coarse k-means buckets first (bounded memory), then agglomerative inside each.
    buckets = KMeans(n_clusters=max(2, len(x) // 1500), n_init=4, random_state=seed).fit_predict(x)
    out, offset = np.full(len(x), -1), 0
    for b in np.unique(buckets):
        idx = np.where(buckets == b)[0]
        sub = _cluster(x[idx], distance, seed)
        out[idx] = sub + offset
        offset += int(sub.max()) + 1
    return out


def _author_groups(ctx, ids: List[str]) -> int:
    """Number of author-disjoint groups (greedy): two papers sharing an author id count once."""
    groups: List[set] = []
    for aid in ids:
        r = ctx.paper_row(aid)
        authors = set(str(r["author_ids"]).split()) if r is not None and str(r.get("author_ids", "")) else {aid}
        for g in groups:
            if g & authors:
                g |= authors
                break
        else:
            groups.append(set(authors))
    return len(groups)


def detect_stated(ctx) -> List[GapCandidate]:
    g = ctx.cfg["gaps"]
    P = ctx.problems
    if P is None or len(P) < int(g["min_support"]) or ctx.problem_vectors is None:
        return []
    X = ctx.problem_vectors[P["vrow"].to_numpy()]
    labels = _cluster(X, float(g["problem_cluster_distance"]), int(ctx.cfg["seed"]))
    dates = pd.to_datetime(P["arxiv_id"].map(ctx.papers.set_index("arxiv_id")["date"]))
    out: List[GapCandidate] = []
    for c in np.unique(labels):
        rows = P[labels == c]
        papers = list(dict.fromkeys(rows["arxiv_id"].astype(str)))
        if len(papers) < int(g["min_support"]):
            continue
        cx = X[labels == c]
        centroid = cx.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        medoid = rows.iloc[int(np.argmax(cx @ centroid))]
        d = dates[labels == c]
        first, last = d.min(), d.max()
        months_open = (ctx.ref_date - first).days / 30.4
        recent = int((d >= ctx.ref_date - pd.DateOffset(months=12)).sum())
        long_standing = months_open >= int(g["longstanding_min_months"]) and recent >= 1
        # Later papers that may address it (collection, scope only): similar abstract, dated after `first`.
        later = (pd.to_datetime(ctx.papers["date"]) > first).to_numpy() & ~ctx.papers["arxiv_id"].isin(papers).to_numpy()
        sims = ctx.vectors @ centroid
        cand = np.where(later & (sims >= float(g["addressed_sim"])))[0]
        attempts = cand[np.argsort(-sims[cand])][:3]
        ev = []
        best = rows.sort_values(["specificity", "verified_ratio"], ascending=False).drop_duplicates("arxiv_id")
        for r in best.head(5).itertuples():
            ev.append(evidence_for(ctx, r.arxiv_id, "states_problem", quote=r.quote,
                                   section=_SECTION_NAMES.get(r.section_kind, r.section_kind), anchor=r.anchor))
        for j in attempts:
            ev.append(evidence_for(ctx, str(ctx.papers.iloc[int(j)]["arxiv_id"]), "related_attempt",
                                   note=f"similar topic (similarity {float(sims[j]):.2f}), published later"))
        support = len(papers)
        out.append(GapCandidate(
            type="long_standing_problem" if long_standing else "stated_open_problem",
            scope=ctx.scope, key_terms={"problem": str(medoid["statement"])[:120]},
            seed_statement=str(medoid["statement"]), evidence=number_evidence(ev),
            stats={"support": support, "author_groups": _author_groups(ctx, papers),
                   "first_seen": f"{first:%Y-%m}", "last_seen": f"{last:%Y-%m}", "recent_support": recent,
                   "months_open": round(months_open, 1), "related_attempts": int(len(attempts)),
                   "problem_types": rows["problem_type"].value_counts().to_dict()},
            specificity=float(rows["specificity"].mean()) / 3.0,
            strength=float(1 - np.exp(-support / 3.0)),
            problem_ids=rows["problem_id"].tolist()))
    return sorted(out, key=lambda c: -c.strength)
```

---

## 7. Detector: unexplored combinations: `gaps/detectors/combinatorial.py`

Statistics: for value pair *(a, b)* with marginals `r_a` (papers with *a*) and `c_b` (papers with *b*) among `N` papers that have both facet types, the expected co-occurrence under independence is `E = r_a · c_b / N`. With observed `O ≤ max_observed`, `p = PoissonCDF(O; E)`. A small `p` means the pair is rarer than chance. **Plausibility guard:** with `require_bridge`, a candidate needs a *bridge*: *a* is used for a task similar to *b* (cosine ≥ `min_bridge_sim` between value vectors), or *b* is approached with a method similar to *a*. Without a bridge it's usually two unrelated communities, not a gap.

```python
"""Method x task (and similar) holes that are statistically unexpected AND plausibly bridgeable."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..evidence import evidence_for, number_evidence
from ..model import GapCandidate

_LABEL = {"method": "method", "task": "task", "domain": "domain", "language": "language", "modality": "modality"}


def _incidence(F: pd.DataFrame, facet: str, top: int, exclude_baseline: bool) -> Tuple[List[str], Dict[str, set]]:
    sub = F[F["facet"] == facet]
    if exclude_baseline and "role" in sub:
        sub = sub[sub["role"] != "baseline"]
    sub = sub[~sub["value"].str.lower().isin({"general", "other", ""})]
    counts = sub.groupby("value")["arxiv_id"].nunique().sort_values(ascending=False).head(top)
    papers = {v: set(sub.loc[sub["value"] == v, "arxiv_id"]) for v in counts.index}
    return list(counts.index), papers


def _bridge(ctx, facet: str, target: str, partners: List[str]) -> Tuple[str, float]:
    vv = ctx.value_vectors.get(facet, {})
    if target not in vv:
        return "", 0.0
    best, s = "", 0.0
    for p in partners:
        if p in vv and p != target:
            sim = float(vv[target] @ vv[p])
            if sim > s:
                best, s = p, sim
    return best, s


def detect_combinatorial(ctx) -> List[GapCandidate]:
    g = ctx.cfg["gaps"]
    out: List[GapCandidate] = []
    for pair in g["combo_pairs"]:
        fa, fb = pair.split(":")
        A_vals, A = _incidence(ctx.facets, fa, int(g["combo_top_values"]), fa == "method")
        B_vals, B = _incidence(ctx.facets, fb, int(g["combo_top_values"]), fb == "method")
        has_a = set().union(*A.values()) if A else set()
        has_b = set().union(*B.values()) if B else set()
        universe = has_a & has_b
        N = len(universe)
        if N < 10:
            continue
        for a in A_vals:
            ra = len(A[a] & universe)
            if ra < int(g["min_marginal"]):
                continue
            used_with = [b for b in B_vals if A[a] & B[b]]             # b' that a is already used for
            for b in B_vals:
                cb = len(B[b] & universe)
                if cb < int(g["min_marginal"]) or a.lower() in b.lower() or b.lower() in a.lower():
                    continue
                obs = len(A[a] & B[b])
                exp = ra * cb / N
                if obs > int(g["max_observed"]) or exp < float(g["min_expected"]):
                    continue
                p = float(poisson.cdf(obs, exp))
                if p > float(g["max_p"]):
                    continue
                b_sim_name, b_sim = _bridge(ctx, fb, b, used_with)
                approaches_b = [x for x in A_vals if A[x] & B[b]]
                a_sim_name, a_sim = _bridge(ctx, fa, a, approaches_b)
                has_bridge = max(b_sim, a_sim) >= float(g["min_bridge_sim"])
                if g["require_bridge"] and not has_bridge:
                    continue
                ev = []
                if b_sim_name:
                    for pid in sorted(A[a] & B[b_sim_name])[:2]:
                        ev.append(evidence_for(ctx, pid, "bridge", note=f"applies {a} to {b_sim_name}"))
                for pid in sorted(A[a] - B[b])[:2]:
                    ev.append(evidence_for(ctx, pid, "method_use", note=f"uses {a}"))
                top_b = ctx.papers[ctx.papers["arxiv_id"].isin(B[b])].sort_values("cited_by_count", ascending=False)
                for pid in top_b["arxiv_id"].head(3):
                    ev.append(evidence_for(ctx, str(pid), "task_use", note=f"works on {b}"))
                bridge_txt = (f"{a} is already used for {b_sim_name}" if b_sim_name else
                              f"{b} is studied with the related {a_sim_name}")
                out.append(GapCandidate(
                    type="unexplored_combination", scope=ctx.scope, key_terms={fa: a, fb: b},
                    seed_statement=(f"No paper in this collection uses {a} for {b}, although {bridge_txt}."),
                    evidence=number_evidence(ev),
                    stats={"pair": pair, "observed": obs, "expected": round(exp, 2), "p_value": round(p, 4),
                           f"n_{fa}": ra, f"n_{fb}": cb, "n_papers": N, "bridge": b_sim_name or a_sim_name,
                           "bridge_sim": round(max(a_sim, b_sim), 3)},
                    specificity=0.8, strength=float(min(1.0, (1 - p) * min(1.0, exp / 10)))))
    out.sort(key=lambda c: (c.stats["p_value"], -c.stats["expected"]))
    return out
```

---

## 8. Detector: evaluation gaps: `gaps/detectors/evaluation.py`

For each method value with ≥ `eval_min_papers` papers (role ≠ baseline), aggregate:
- `datasets`: distinct canonical datasets across those papers,
- `profile` shares from `paper_profile` (`eval_real_world`, `eval_robustness`, `eval_human_eval`, `eval_multilingual`, `eval_low_resource`), using only **known** values (not None). A share is computed only when known for ≥ `eval_min_known` of papers,
- `languages`: union of language facets.

Triggers (one candidate per method × trigger):

| Trigger | Condition | Seed statement |
|---|---|---|
| `narrow_benchmarks` | ≤ `eval_max_datasets` distinct datasets | "{m} has been evaluated on only {k} benchmark(s) ({list}) across {n} papers." |
| `no_real_world` | real-world share == 0 (known) | "None of the {n} papers on {m} evaluates it on real-world/deployment data." |
| `no_robustness` | robustness share == 0 (known) | "Robustness of {m} under distribution shift or adversarial conditions is untested here." |
| `english_only` | all known languages == {English} and modality includes text | "{m} has only been evaluated on English data." |
| `no_efficiency` | efficiency share == 0 (known) and compute ≥ multi_gpu common | "The cost/latency of {m} is not reported." |

Evidence: up to 5 papers with `note="evaluated on: …"` (their dataset list) and role `narrow_eval`. Strength = `min(1, n/10)`. Specificity 0.7.

---

## 9. Detector: unsettled questions: `gaps/detectors/unsettled.py`

- Map each finding's `task`/`dataset`/`metric` to canonical values. Use `facet_vocab` aliases (raw→canonical) where they exist, else `extract.canonical.normalize_value`.
- Group by `(task, dataset, metric)` and keep groups with ≥ `unsettled_min_papers` distinct papers.
- Conflict: two findings in a group whose `compared_to` are similar (`rapidfuzz.fuzz.token_set_ratio ≥ 70`, or one names the other's method) and whose directions disagree (`improves` vs `no_effect`/`worsens`, or `mixed`).
- Candidate per conflicting group. Evidence role `conflicting_result`, with `note = "Reported (paraphrased): " + claim`. Findings are paraphrases, never shown as quotes.
- These go through the LLM **conflict judge** (§10), because direction labels are noisy.

---

## 10. Detector: missing bridges: `gaps/detectors/bridge.py`

Two levels:
- **Offline:** between **areas** (global pass, scope `{"kind": "global"}`).
- **Online:** between **workspace sub-topics** (`ctx.sub_labels`).

Algorithm for groups G (areas or sub-topics):
1. Centroid per group (unit). Pairwise cosine; candidate pairs above the `bridge_sim_quantile` of all pair similarities.
2. Map OpenAlex `openalex_id` → group for papers in the corpus. For each group, `refs_out[g]` = references from its papers that point to *any* corpus paper; `cross[g→h]` = those pointing into *h*.
3. Expected cross-citations from g to h ∝ the share of corpus papers in h: `exp = refs_out[g] · |h| / N_corpus_with_openalex`. `ratio = (cross[g→h] + cross[h→g]) / (exp_gh + exp_hg)`.
4. Require reference coverage ≥ `bridge_min_ref_coverage` in both groups. Candidate if `ratio ≤ bridge_max_ratio`.
5. Evidence: the 2 most central papers of each side (`side_a`, `side_b`). Seed: "{A} and {B} study closely related ideas (similarity {s:.2f}) but cite each other {ratio:.0%} as often as expected."

---

## 11. Judges and synthesis (LLM): `gaps/prompts.py`, `gaps/synthesize.py`

### 11.1 Schemas

```python
class ComboJudgement(BaseModel):
    meaningful: bool          # would applying A to B be a sensible research question?
    trivial: bool             # obvious/uninteresting, or a plain engineering exercise
    inapplicable_reason: str  # "" if applicable
    reason: str               # one sentence


class ConflictJudgement(BaseModel):
    is_conflict: bool
    explanation: str
    possible_reasons: List[str]   # e.g. different splits, model sizes, metrics


class AddressedJudgement(BaseModel):
    verdict: Literal["addresses", "partially", "unrelated"]
    reason: str
    quote: str                    # exact supporting sentence from the abstract, or ""


class GapDraft(BaseModel):
    reject: bool
    reject_reason: str
    title: str
    statement: str
    why_it_matters: str
    why_it_might_not_be_a_gap: List[str]
    first_steps: List[str]
    skills: List[str]
    data_needs: str
    compute: Literal["laptop", "single_gpu", "multi_gpu", "large_cluster", "unknown"]
    feasibility: Literal["low", "medium", "high"]
    beginner_friendly: bool
    evidence_used: List[str]
    novelty_queries: List[str]
```

Give each `str` field a clipping validator (title 90 chars, statement 400, why_it_matters 300; list items 200 each; lists capped at 5) following plan 01's rule.

### 11.2 Prompts (register; versions start at `"1"`)

**`judge_combination`** (task `judge`):
- system: "You assess whether combining a method and a research problem is a sensible, non-trivial research question. Use general CS knowledge only to judge applicability, not to claim whether it has been done."
- user: `Method/value A: $a ($fa)\nProblem/value B: $b ($fb)\nBridge: $bridge\nEvidence:\n$evidence`

**`judge_conflict`** (task `judge`): the conflicting findings with paper titles. "Decide if these results genuinely conflict on the same question, and list plausible reasons."

**`judge_addressed`** (task `judge`):
- system: "Decide whether the candidate paper already addresses the stated research gap. 'addresses' = it directly tackles the gap; 'partially' = covers part of it or a close variant; 'unrelated' otherwise. Quote one exact sentence from the abstract that supports your verdict, or leave quote empty."
- user: `<evidence>\nGap: $statement\n</evidence>\n<abstract>\nTitle: $title\n$abstract\n</abstract>`

**`synthesize_gap`** (task `synthesize_gap`). The system prompt must say:

```text
You write one "Gap Card" for a beginner researcher from a gap CANDIDATE found by an automatic detector.
Use ONLY the candidate's evidence and statistics. Rules:
1. If the candidate is not a real, meaningful research gap (trivial, nonsensical, already a standard
   practice, or not supported by the evidence), set reject=true and explain in reject_reason.
2. statement: 1-2 sentences, specific and testable. Say "in the papers analysed here", never
   "no one has ever". Name the method/task/setting explicitly.
3. Mention only methods, datasets, tasks and names that appear in the evidence or key terms.
   Do not introduce new named entities.
4. why_it_might_not_be_a_gap: 2-4 concrete, honest reasons (e.g. work may exist outside arXiv,
   in industry, under different names; the combination may be infeasible because ...).
5. first_steps: 3 concrete actions a beginner can start this month, referencing evidence ids
   (e.g. "Reproduce E2's baseline on ...").
6. evidence_used: the evidence ids you relied on. novelty_queries: 2-4 short search queries
   another system will use to look for papers that already did this.
7. feasibility/compute: judge from the evidence (datasets public? code available? model scale?).
```

User template:

```text
Gap type: $type_label - $type_help
Scope: $scope_label
Key terms: $key_terms
Detector statistics:
$stats
<evidence>
$evidence
</evidence>
Detector's draft statement (may be clumsy): $seed
```

Evidence lines: `E1 [states_problem] (2024) "Title" | section: Limitations | quote: "…"` or `| note: …`.

Register `judge_combination`, `judge_conflict` and `judge_addressed` with `packable=True` and call them through `llm.structured_many("judge", ...)`. Keep `synthesize_gap` unpacked.

Fake handlers (register for all four prompts): `judge_combination` → meaningful=true, trivial=false. `judge_conflict` → is_conflict=true. `judge_addressed` → "unrelated" unless the abstract contains the word "ADDRESSES_GAP" (lets tests plant a hit). `synthesize_gap` → title from key terms, statement = seed, two generic caveats, three generic steps, `evidence_used` = all ids, `novelty_queries` = [key terms joined].

### 11.3 `synthesize.py`

```python
def template_card(c: GapCandidate) -> GapDraft:
    """No-LLM fallback: honest, generic wording per type."""


def synthesize(cands: List[GapCandidate], llm, cfg, offline: bool) -> List[Tuple[GapCandidate, GapDraft, Dict]]:
    """Returns (candidate, draft, generated_by). Uses llm.structured_many("synthesize_gap", ...) (pack_size 1:
    one card per request keeps quality high). Per item: LLMError -> template_card. draft.reject -> dropped (counted).
    After the LLM: verify_text(draft, candidate); on failure retry ONCE synchronously with an extra
    line in the seed: 'Previous draft named entities not in evidence: X, Y. Remove them.'; if it fails
    again -> template_card."""
```

---

## 12. Text verification: `gaps/verify.py`

```python
"""Guards between the LLM and the user."""
from __future__ import annotations

import re
from typing import List, Set

from .model import GapCandidate

_ENTITY = re.compile(r"\b(?:[A-Z][a-z]*[A-Z][A-Za-z0-9\-]*|[A-Z]{2,}[A-Za-z0-9\-]*|[A-Z][a-z]+-\d+[A-Za-z]*)\b")
_ALLOWED = {"AI", "ML", "NLP", "LLM", "LLMs", "GPU", "GPUs", "CPU", "API", "OOD", "SOTA", "E1", "E2", "E3",
            "E4", "E5", "E6", "E7", "E8", "US", "UK", "EU"}


def candidate_vocab(c: GapCandidate) -> str:
    parts = [c.seed_statement, " ".join(c.key_terms.values()), str(c.scope.get("label", ""))]
    for e in c.evidence:
        parts += [e.title, e.quote, e.note]
    return " ".join(parts).lower()


def foreign_entities(text: str, vocab: str) -> List[str]:
    """Acronym/CamelCase/model-number tokens in ``text`` absent from the candidate's own material."""
    out: Set[str] = set()
    for m in _ENTITY.finditer(text):
        tok = m.group(0)
        if tok in _ALLOWED or tok.lower() in vocab:
            continue
        out.add(tok)
    return sorted(out)


def verify_text(draft, c: GapCandidate) -> List[str]:
    """Problems found (empty list = OK)."""
    vocab = candidate_vocab(c)
    issues = []
    ids = {e.eid for e in c.evidence}
    bad_ids = [i for i in draft.evidence_used if i not in ids]
    if bad_ids:
        issues.append(f"unknown evidence ids {bad_ids}")
    text = " ".join([draft.title, draft.statement, draft.why_it_matters, *draft.first_steps])
    foreign = foreign_entities(text, vocab)
    if foreign:
        issues.append("entities not in evidence: " + ", ".join(foreign))
    if not draft.why_it_might_not_be_a_gap:
        issues.append("missing why_it_might_not_be_a_gap")
    if re.search(r"\b(no one|nobody) has ever\b|\bfirst ever\b", draft.statement, re.I):
        issues.append("overclaiming language")
    return issues
```

Evidence integrity: before synthesis, assert every `Evidence.quote` exists in `bundle.problems.quote` for that `paper_id` (they come from there). Drop any that don't, which would be a bug signal; log it.

---

## 13. Novelty check: `gaps/novelty.py`

```python
def novelty_check(card: GapCard, ctx: GapContext, embedder, llm, cfg, searcher=None,
                  live: bool = False) -> NoveltyCheck:
    """
    1. statement vector = embedder.transform([card.statement]) (unit).
    2. Collection candidates: papers dated >= ref_date - novelty_recent_months, excluding evidence papers,
       top `novelty_collection_top` by cosine against ctx.all_vectors (the whole bundle, not just scope).
    3. Live candidates (live=True and searcher): for q in (card's novelty_queries or key terms)[:novelty_live_queries]:
          res = searcher.run(q, pages=1, live=True)   # existing LiveSearcher
       keep results with a date in the recent window; score by cosine when a vector exists
       (bundle_index -> bundle vector; live store row -> store vector) else by word overlap.
    4. Judge the top `novelty_top_judge` by score with prompt `judge_addressed` (task judge) via
       llm.structured_many (packed: all candidates of one card in one request). Verify each returned quote against the candidate abstract with
       scholargrid.extract.verify.locate(min_ratio=90); blank it if not found.
       Without LLM: heuristic verdict 'partially' if cosine >= 0.85 else 'unrelated'
       (never 'addresses' without an LLM), method='heuristic'.
    5. status = already_addressed if any 'addresses'; partially_explored if any 'partially'; else novel.
       scope = 'collection+live' if live else 'collection'; checked_at = now.
    """
```

- Offline stage: `live=False` (cheap, deterministic). The card shows "Checked against the collection only".
- Online: `live=True` for newly generated workspace cards, plus a **"Check novelty now"** button on any card whose check is collection-only or older than `novelty_ttl_days`.
- Cache novelty results by `(card.id, scope)` in the LLM disk cache (automatic through `structured`) and in the workspace gap cache.

---

## 14. Scoring and confidence: `gaps/scoring.py`

```python
def _has_code(ctx: GapContext, ids: list[str]) -> bool:
    p = ctx.profiles
    if p is None or p.empty or "code_url" not in p.columns:
        return False
    return bool((p[p["arxiv_id"].isin(ids)]["code_url"].fillna("").astype(str) != "").any())


def score(card: GapCard, cand: GapCandidate, ctx: GapContext, cfg) -> Scores:
    w = cfg["gaps"]["weights"]
    s = cand.stats
    if card.type in ("stated_open_problem", "long_standing_problem"):
        evidence = 1 - math.exp(-s["support"] / 3) * (1.0 if s.get("author_groups", 1) > 1 else 1.3)
        ev_txt = f"{s['support']} papers ({s.get('author_groups', '?')} independent author groups) state it"
    elif card.type == "unexplored_combination":
        evidence = (1 - s["p_value"]) * min(1.0, s["expected"] / 8)
        ev_txt = f"expected ~{s['expected']} papers combining them, found {s['observed']}"
    elif card.type == "evaluation_gap":
        evidence = min(1.0, s.get("n_papers", 0) / 10)
        ev_txt = f"{s.get('n_papers')} papers on the method"
    elif card.type == "unsettled_question":
        evidence = min(1.0, s.get("n_conflicting", 0) / 4)
        ev_txt = f"{s.get('n_conflicting')} conflicting results"
    else:
        evidence = 1 - min(1.0, s.get("ratio", 1.0))
        ev_txt = f"cross-citation {s.get('ratio', 0):.0%} of expected"
    evidence = max(0.0, min(1.0, evidence))
    novelty = {"novel": 1.0, "partially_explored": 0.5, "not_checked": 0.6, "already_addressed": 0.0}[card.novelty.status]
    feas = {"high": 0.9, "medium": 0.6, "low": 0.3}[card.feasibility]
    if _has_code(ctx, [e.paper_id for e in card.evidence]):
        feas = min(1.0, feas + 0.1)
    years = [e.year for e in card.evidence if e.year]
    recency = sum(1 for y in years if y >= ctx.ref_date.year - 1) / max(1, len(years))
    spec = max(0.0, min(1.0, cand.specificity))
    total = (w["evidence"] * evidence + w["novelty"] * novelty + w["feasibility"] * feas
             + w["specificity"] * spec + w["recency"] * recency)
    return Scores(evidence=round(evidence, 3), novelty=novelty, feasibility=round(feas, 3), specificity=round(spec, 3),
                  recency=round(recency, 3), total=round(total, 3),
                  explain={"evidence": ev_txt, "novelty": card.novelty.status.replace("_", " "),
                           "feasibility": f"{card.feasibility} (from evidence: data/code/compute)",
                           "specificity": f"{spec:.0%} concrete", "recency": f"{recency:.0%} of evidence from the last 2 years"})


def confidence(card: GapCard) -> str:
    s = card.scores
    live = card.novelty.scope == "collection+live"
    if s.evidence >= 0.7 and card.novelty.status == "novel" and live:
        return "high"
    if s.evidence >= 0.4 and card.novelty.status in ("novel", "partially_explored", "not_checked"):
        return "medium"
    return "low"
```

---

## 15. Engine, stage, online mode, page

### 15.1 `gaps/engine.py`

```python
DETECTORS = {"stated": detect_stated, "combinatorial": detect_combinatorial, "evaluation": detect_evaluation,
             "unsettled": detect_unsettled, "bridge": detect_bridge}


def candidates_for(ctx) -> List[GapCandidate]:
    out = []
    for name in ctx.cfg["gaps"]["detectors"]:
        try:
            out += DETECTORS[name](ctx)
        except Exception as exc:          # one detector must never break the others
            log.exception("Detector %s failed on %s: %s", name, ctx.scope, exc)
    seen, uniq = set(), []
    for c in sorted(out, key=lambda c: -c.strength):
        if c.id not in seen:
            seen.add(c.id)
            uniq.append(c)
    return uniq[: int(ctx.cfg["gaps"]["max_candidates_per_scope"])]


def finalize(cands, ctx, embedder, llm, cfg, *, offline: bool, searcher=None, live: bool = False,
             max_cards: int | None = None) -> Tuple[List[GapCard], Dict]:
    """judge -> synthesize -> verify -> novelty -> score -> filter -> sort. Returns (cards, stats)."""
```

Stats returned: `candidates`, `judged_out`, `rejected_by_llm`, `text_fallbacks`, `already_addressed`, `cards`, `by_type`, `requests_used` (difference of `llm.ledger.requests` before and after), `quota_stopped`.

### 15.2 Offline stage `gaps`

```python
STAGES = [..., "analyze", "foundations", "gaps", "validate", "publish"]
_PROGRESS["gaps"] = (0.86, "Detecting open problems…")


def stage_gaps(ctx: Context) -> None:
    from .gaps.engine import run_offline_gaps

    cfg = ctx.cfg
    with stage_timer("gaps", ctx.timings) as info:
        stats = run_offline_gaps(cfg, cfg.processed_dir) if cfg["gaps"]["enabled"] else {}
        info.update(**{k: v for k, v in stats.items() if isinstance(v, (int, float))})
    # merge into meta.json exactly like stage_foundations
```

`run_offline_gaps`:
1. `b = load_bundle(cfg, directory=cfg.processed_dir)`.
2. For each area with ≥ 30 papers: `build_context(b, idx, {"kind": "area", "id": cid, "label": area_name}, cfg, b.embedder)` → `candidates_for` → collect.
3. A global bridge pass across areas.
4. `finalize(..., offline=True, live=False)`. The LLM judges and synthesis run across all areas in **priority order** (areas with the most verified stated problems first): collect all items first, then call `llm.structured_many` per prompt. Judges are packed (`judge_*` prompts are `packable=True`; `llm.tasks.judge.pack_size` = 5), so 5 candidate checks cost 1 request.
5. Write `gaps.json`: `{"generated_at", "version": 1, "areas": {cid: [card.model_dump()]}, "global": [...], "stats": {...}}`.

Add `GAPS_JSON = "gaps.json"` and a `Bundle.gaps: Dict` field (default `{}`), loaded optionally.

**Quota control:** before the LLM steps, log the expected number of requests (`candidates × (1 synthesis + ⌈judges/5⌉)`) next to `python -m scholargrid.llm quota`. On `QuotaExhausted` or `BudgetExceeded`, write the remaining candidates as `template_card`s marked `generated_by.synthesized=false`. The next pipeline run re-synthesizes them: LLM results are cached per candidate, so finished cards cost nothing again.

### 15.3 Online workspace mode: `gaps/workspace.py`

```python
def gaps_for_workspace(ws: TopicWorkspace, bundle, searcher, cfg, llm, progress=None) -> Dict:
    """Cards for the topic: (a) detectors on the whole workspace + per sub-topic (bundle papers only),
    bridges across sub-topics; (b) relevant offline area cards (areas covering >= 20% of the workspace's
    collection papers, card statement cosine >= 0.35 to the query vector) labelled 'From the wider area'.
    New cards: finalize(offline=False, live=True, max_cards=online_max_new_cards).
    Cache: data/workspaces/<ws.key>.gaps.json  {"cards": [...], "area_cards": [...], "stats": {...}}"""
```

Workspace papers have sub-topic labels. Pass `sub_labels` aligned with `idx` to `build_context` for bridges.

### 15.4 Page `app/views/gaps.py` ("Open problems")

Add it to the navigation (`"gaps": st.Page(gaps.render, title="Open problems", icon=":material/explore:", url_path="open-problems")`).

Layout:
1. Guard: needs a workspace (same as Topic). Offer "or browse open problems by area" (select box of areas → shows `bundle.gaps["areas"][cid]`).
2. Header "Open problems in: {query}" plus the honest one-liner: "Each card is built from what authors wrote and what the papers contain. Read the evidence, and treat every card as a lead to verify."
3. **Instant section:** area cards relevant to the topic ("From the wider area").
4. **Generate section:** if there is no cached topic-specific result, a primary button "Find open problems for this topic (about 1–2 minutes)" → `st.status` progress → `gaps_for_workspace`. Cached results show directly.
5. Filters (sidebar-free row): type multiselect (with `TYPE_HELP` tooltips), confidence, feasibility, "Beginner-friendly only", sort (score / novelty / feasibility / newest evidence).
6. **Card renderer** `ui.gap_card(card, key_prefix)`:
   - title; chips for the type (with the plain explanation), confidence, feasibility, "Beginner-friendly",
   - statement in larger text; "Why it matters",
   - **"Why this might not be a gap"**: always visible (the first reason inline, the rest in a small list),
   - novelty line, e.g. "No paper found that already does this (checked 3 days ago: collection + arXiv, OpenAlex, Semantic Scholar)" / "2 related papers found" (linked) / "Checked against the collection only" + **"Check novelty now"** button,
   - score bar with the 5 components; hover text from `scores.explain`,
   - expander **"Evidence ({n} papers)"**: per item, title (link), year, role in words, the quote in a blockquote with the section name, and a "Read in context" deep link (`anchor_url`), or the note,
   - expander **"How to start"**: first steps, skills, data needs, compute,
   - buttons: "Save to project" (plan 08; until then `st.session_state.saved_gaps`), "Make a starter plan" (plan 07), "Not useful" (plan 09 feedback event).
7. Empty states: no candidates ("This topic has few author-stated limitations in the analysed papers; try a broader topic or another sub-topic") and LLM unavailable ("Cards use template wording").
8. Set `j_gaps = True`.

Retire the old Leads from the journey: keep `leads` under More labelled "(legacy)". Plan 10 removes it from navigation.

---

## 16. Tests

### 16.1 Unit tests with hand-built tables (`tests/test_gaps_detectors.py`)

Write a helper `make_ctx(papers_df, facets_df, problems_df, problem_vecs, findings_df, profiles_df, vectors, value_vectors)` that builds a `GapContext` directly, without a bundle.

1. **stated:** 4 papers with near-identical problem vectors (cosine 0.98) and 2 unrelated → one candidate, `support=4`, evidence quotes equal the table quotes, type `long_standing_problem` when the first is ≥ 24 months before `ref_date` and one is recent, else `stated_open_problem`. Below `min_support` → none.
2. **stated related attempts:** a later paper whose vector is close to the centroid appears as a `related_attempt` evidence item.
3. **combinatorial:** construct 40 papers. Method `M` is used for tasks `T1` (12 papers) and the similar `T1b` (value vectors cosine 0.8). Task `T2` (12 papers) is studied with other methods only. `T2` is similar to `T1` (0.7). Expect candidate `(M, T2)` with `observed=0`, `p ≤ 0.05`, and bridge `T1`. With `require_bridge` and all similarities 0.1 → no candidate.
4. **evaluation:** method with 6 papers all on dataset `D1` → `narrow_benchmarks`; with known `eval_real_world=False` for all → `no_real_world`; unknown (None) for most → no `no_real_world` candidate.
5. **unsettled:** two findings on (task, dataset, metric) with directions `improves` vs `no_effect` and similar `compared_to` → one candidate; same direction → none.
6. **bridge:** two sub-groups with close centroids, refs coverage 50%, zero cross-citations → candidate; plenty of cross-citations → none.

### 16.2 Guards (`tests/test_gaps_verify.py`)
- `foreign_entities("Apply GPT-9 to SQuAD", vocab="squad ...")` → `["GPT-9"]`.
- A draft with unknown evidence id `E9` → issue. Overclaiming "no one has ever" → issue.
- The synthesize retry path: a FakeLLM whose first `synthesize_gap` answer contains a foreign entity and whose second doesn't → final draft is the second. If both fail → `template_card` (marked `synthesized=False`).

### 16.3 Novelty (`tests/test_gaps_novelty.py`)
- A planted recent collection paper whose abstract contains `ADDRESSES_GAP` and whose vector is near the statement → status `already_addressed`. The card is dropped by `finalize` (or kept in `already_explored` when `keep_already_addressed`).
- Without LLM (`provider: none`) → `method="heuristic"`, never `already_addressed`.
- The live path with a stub searcher whose `run()` returns a result dict list → hits include it.

### 16.4 Scoring (`tests/test_gaps_scoring.py`)
Monotonicity: more support → higher evidence; `novel` > `partially` > `already`; confidence is `high` only with a live check.

### 16.5 End-to-end
- `tests/test_pipeline.py`: after `run(cfg)`, `load_bundle(cfg).gaps["areas"]` has ≥ 1 card (synthetic planted problems recur). Every card has ≥ 1 evidence item, non-empty `why_it_might_not_be_a_gap`, a novelty status ≠ `not_checked` (collection check ran), and every evidence quote appears verbatim in `bundle.problems.quote`.
- `tests/test_gaps_workspace.py`: build a workspace (plan 05 helper) → `gaps_for_workspace` (stub searcher, FakeLLM) → cards cached; a second call does no LLM calls.
- `tests/test_app.py`: add `"gaps"` to `PAGES`. With a workspace in the session and a cached gaps file, the page renders the card titles and the text "Why this might not be a gap".

---

## 17. Acceptance checks

```bash
ruff check . && mypy && pytest -q
python -m scholargrid.pipeline --config configs/ci.yaml --no-publish
python - <<'EOF'
from scholargrid.config import load_config
from scholargrid.artifacts import load_bundle
cfg = load_config("configs/ci.yaml"); b = load_bundle(cfg, directory=cfg.processed_dir)
cards = [c for cs in b.gaps["areas"].values() for c in cs]
print(len(cards), "cards"); c = cards[0]
print(c["type"], "|", c["title"]); print(c["statement"]); print(c["why_it_might_not_be_a_gap"])
for e in c["evidence"]: print(" ", e["eid"], e["role"], e["title"][:60], "|", e["quote"][:80])
EOF
streamlit run app/streamlit_app.py     # Start → topic → Open problems → generate → inspect evidence
```

**Real-data gate before moving on (manual, required):** on `configs/domain_nlp_dev.yaml` (≥ 1,000 extracted papers: raise `scope.max_papers` to 1,000 and let the `dev-build` workflow from plan 01 §8.2 run on GitHub for a few days of free quota; nothing heavy on the laptop), download the release artifact, open it with `streamlit run`, and generate cards for 3 topics you know. For 10 cards per topic, mark: real and open / already solved / nonsense. Record the counts in `Docs/runbook.md`. If fewer than half are "real and open", tune thresholds (`min_support`, `problem_cluster_distance`, `max_p`, `require_bridge`, `addressed_sim`) and prompts **before** building plan 07. Plan 09 then formalizes this evaluation.

## 18. Pitfalls

- **Precision over recall.** Twelve good cards beat a hundred mediocre ones. Defaults are strict on purpose.
- Combinatorial holes between unrelated communities are the classic false positive. Keep `require_bridge: true` unless you have evidence it hurts.
- Findings are LLM paraphrases. Never render them as quotes (role `conflicting_result` → "Reported (paraphrased)").
- Novelty "novel" means "no match among what was checked". The UI must say what was checked and when.
- Agglomerative clustering is O(n²) memory: the k-means bucketing above 4,000 problems is required for large areas.
- Keep detectors pure and fast; all LLM work happens in `finalize`, so the order and the request count are controlled in one place.
- **Free-tier budget online:** one topic's "Find open problems" costs about `online_max_new_cards × 1` synthesis requests plus ≈ 1 packed judge request per card (≈ 16 requests with the defaults). The server's shared free quota covers only a handful of topics per day, which is why plan 11 makes bring-your-own-key (each visitor's own free Gemini key) the main mode for public use. Lower `online_max_new_cards` if needed.
