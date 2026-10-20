# Plan 04: Foundations and reading paths (new stage `foundations`)

> **Prerequisites:** plans 01–03 are merged. OpenAlex enrichment (`enrich` stage, existing) fills `references` (space-separated OpenAlex work IDs such as `W2741809807`) for matched papers.
> **Produces:**
> 1. a **citation backbone** per area: works that many papers in the area cite, including **pre-2023 foundations that are not in the corpus**, with metadata fetched from OpenAlex and cached,
> 2. **survey detection**,
> 3. an ordered **reading path** per area (surveys → foundations → key recent → frontier), each item with a one-line reason,
> 4. a reusable `reading_path()` for any subset of papers (used online by the topic workspace in plan 05),
> 5. a "Read first" panel on the existing Areas detail page.
>
> **Why:** the corpus starts in 2021–2023, so a beginner never sees the papers everything builds on. "What do I read first?" is the second question every newcomer asks.

---

## 1. Prompt to paste into Cursor

```text
Implement plan file 04_foundations_reading_path.md. Follow 00_INDEX_READ_FIRST.md and plan 01's LLM
contract. Order: §3 config → §4 survey detection → §5 WorkLookup (OpenAlex by ID, cached) →
§6 synthetic references for offline runs → §7 backbone scoring → §8 reading_path() → §9 optional LLM
reasons → §10 stage + bundle files + loading → §11 Areas page panel → §12 tests. Do not add network
calls to tests. Finish with §13.
```

---

## 2. Concepts

**Co-citation backbone.** For an area *A* with `n_A` papers that have references:
- `freq_A(W)` = number of papers in *A* that cite work *W*
- `share_A(W) = freq_A(W) / n_A`
- `share_all(W) = freq_all(W) / n_all`, over all papers with references
- `lift(W) = (share_A(W) + ε) / (share_all(W) + ε)`. A high lift means *W* is specific to this area, not a universal citation like an optimizer paper.
- `score(W) = freq_A(W) · log(1 + lift(W))`

Keep *W* if `freq_A ≥ min_citing_papers` **and** `share_A ≥ min_share`, then take the top `top_per_area` by score.

**Reading path steps.**

| Step | Source | Rank by | Reason template |
|---|---|---|---|
| `survey` | in-area surveys (corpus) + backbone works that are reviews | citations / (age + 1), recent first | "Survey from {year} covering this area ({cites} citations)." |
| `foundation` | backbone works (any year) | score; then shown **oldest first** | "Cited by {freq} of the {n} papers here, a shared foundation." |
| `recent` | corpus papers from the last `recent_months` | citation velocity, else citations per month | "One of the most-cited recent papers here ({rate:.1f} citations/month)." |
| `frontier` | corpus papers from the last `frontier_months` near the area centroid | similarity to centroid; prefer papers with stated open problems | "Recent ({month}) and central to this area; it lists open problems you can build on." |

Items are de-duplicated across steps (a paper appears once, in its earliest step).

---

## 3. Config

### 3.1 `scholargrid/schema.py`

```python
class Foundations(_Section):
    enabled: bool = True
    min_citing_papers: int = Field(default=3, ge=1)
    min_share: float = Field(default=0.03, ge=0, le=1)
    top_per_area: int = Field(default=30, ge=1)
    lift_epsilon: float = Field(default=0.002, gt=0)
    lookup_batch: int = Field(default=50, ge=1, le=100)      # OpenAlex OR-filter allows up to 100 values
    lookup_ttl_days: int = Field(default=90, ge=1)
    lookup_timeout_s: float = Field(default=20.0, gt=0)
    online_deadline_s: float = Field(default=6.0, gt=0)
    path_sizes: Dict[str, int] = Field(default_factory=lambda: {"survey": 3, "foundation": 6,
                                                                "recent": 6, "frontier": 5})
    recent_months: int = Field(default=24, ge=1)
    frontier_months: int = Field(default=6, ge=1)
    llm_reasons: bool = True
```

Add `foundations: Foundations = Field(default_factory=Foundations)` to `ConfigModel` and the YAML block to `configs/config.yaml` with the same values and short comments.

---

## 4. Survey detection: `scholargrid/foundations/surveys.py`

```python
"""Is this paper a survey / review / tutorial? Cheap and transparent; run at load time too."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_TITLE = re.compile(r"\b(survey|review|overview|tutorial|primer|systematic\s+(literature\s+)?review|"
                    r"sok\b|systemati[sz]ation\s+of\s+knowledge|taxonomy|roadmap|state[- ]of[- ]the[- ]art\s+in|"
                    r"a\s+decade\s+of|lessons\s+learned|perspectives?\s+on)\b", re.I)
_ABSTRACT = re.compile(r"^(this|in\s+this)\s+(survey|review|tutorial)|\bwe\s+(survey|review)\s+(the\s+)?"
                       r"(recent\s+)?(literature|work|advances|progress|methods)", re.I)
_NEG = re.compile(r"\bpeer[- ]review|review\s+(generation|comments?|summari[sz]ation)|code\s+review|"
                  r"product\s+review|reviewer", re.I)


def is_survey(title: str, abstract: str = "", openalex_type: str = "") -> bool:
    if openalex_type == "review":
        return True
    t = str(title or "")
    if _NEG.search(t):
        return False
    return bool(_TITLE.search(t) or _ABSTRACT.search(str(abstract or "")[:400]))


def survey_mask(df: pd.DataFrame) -> np.ndarray:
    return np.array([is_survey(t, a) for t, a in zip(df["title"].astype(str), df["abstract"].astype(str))])
```

The package layout is `scholargrid/foundations/__init__.py` (exports), `surveys.py`, `lookup.py`, `backbone.py`, `path.py`, `prompts.py` and `synthetic.py`.

---

## 5. OpenAlex lookup by work ID: `scholargrid/foundations/lookup.py`

- Endpoint: `GET https://api.openalex.org/works?filter=ids.openalex:W1|W2|…&per-page=100&select=…`. Use up to `lookup_batch` IDs per request.
- Polite pool: add `mailto=$OPENALEX_MAILTO`. Reuse the existing `RateLimiter(0.1)` from `scholargrid/sources/openalex.py` (`LIMITER`) and `http_get` from `scholargrid/sources/__init__.py`.
- Abstracts come as an inverted index; reuse `scholargrid.sources.openalex.abstract_from_index`.
- arXiv IDs: reuse `scholargrid.enrich.work_arxiv_ids` (it reads the DOI and every location).
- Cache: `data/raw/openalex_works_cache.parquet`, one row per work, refreshed after `lookup_ttl_days`.

```python
"""Metadata for arbitrary OpenAlex works (foundations are often outside the corpus)."""
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..config import Config
from ..utils import get_logger

log = get_logger("foundations.lookup")
API = "https://api.openalex.org/works"
SELECT = ("id,doi,title,publication_year,cited_by_count,type,abstract_inverted_index,primary_location,"
          "locations,authorships,ids")
COLUMNS = ["openalex_id", "title", "year", "cited_by_count", "doi", "arxiv_id", "type", "venue",
           "abstract", "authors", "url", "fetched_at"]


def _row(w: Dict) -> Dict:
    from ..enrich import work_arxiv_ids
    from ..sources.openalex import abstract_from_index

    loc = (w.get("primary_location") or {})
    src = loc.get("source") or {}
    arxiv = sorted(work_arxiv_ids(w))
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    names = [((a.get("author") or {}).get("display_name") or "") for a in (w.get("authorships") or [])][:6]
    oid = (w.get("id") or "").rsplit("/", 1)[-1]
    return {"openalex_id": oid, "title": w.get("title") or "", "year": w.get("publication_year"),
            "cited_by_count": int(w.get("cited_by_count") or 0), "doi": doi,
            "arxiv_id": arxiv[0] if arxiv else "", "type": w.get("type") or "",
            "venue": src.get("display_name") or "", "abstract": abstract_from_index(w.get("abstract_inverted_index")),
            "authors": "; ".join(n for n in names if n),
            "url": (f"https://arxiv.org/abs/{arxiv[0]}" if arxiv else
                    (f"https://doi.org/{doi}" if doi else f"https://openalex.org/{oid}")),
            "fetched_at": time.time()}


class WorkLookup:
    def __init__(self, cfg: Config, synthetic: bool = False):
        self.cfg = cfg
        self.f = cfg["foundations"]
        self.path = os.path.join(cfg.raw_dir, "openalex_works_cache.parquet")
        self.synthetic = synthetic
        self.df = pd.read_parquet(self.path) if os.path.exists(self.path) else pd.DataFrame(columns=COLUMNS)

    def get(self, ids: Iterable[str], deadline: Optional[float] = None) -> pd.DataFrame:
        """Rows for ``ids`` (cached or fetched). Missing works are simply absent."""
        ids = [i for i in dict.fromkeys(ids) if i]
        if self.synthetic:
            from .synthetic import synthetic_works

            return synthetic_works(ids)
        ttl = time.time() - float(self.f["lookup_ttl_days"]) * 86400
        have = self.df[self.df["openalex_id"].isin(ids)]
        fresh = set(have.loc[have["fetched_at"] >= ttl, "openalex_id"])
        todo = [i for i in ids if i not in fresh]
        if todo:
            self._fetch(todo, deadline)
        return self.df[self.df["openalex_id"].isin(ids)].reset_index(drop=True)

    def _fetch(self, ids: List[str], deadline: Optional[float]) -> None:
        from ..sources import RateLimited, SourceUnavailable, http_get
        from ..sources.openalex import LIMITER

        mailto = os.environ.get("OPENALEX_MAILTO") or self.cfg["enrich"].get("mailto") or ""
        rows: List[Dict] = []
        size = int(self.f["lookup_batch"])
        for i in range(0, len(ids), size):
            if deadline is not None and time.time() > deadline:
                log.info("Work lookup deadline reached; %d works left for later.", len(ids) - i)
                break
            chunk = ids[i:i + size]
            params = {"filter": "ids.openalex:" + "|".join(chunk), "per-page": str(len(chunk)), "select": SELECT}
            if mailto:
                params["mailto"] = mailto
            try:
                LIMITER.acquire(deadline)
                data = http_get(API, params, float(self.f["lookup_timeout_s"])).json()
            except (RateLimited, SourceUnavailable) as exc:
                log.warning("OpenAlex lookup paused (%s).", exc)
                break
            except Exception as exc:
                log.warning("OpenAlex lookup batch failed (%s); continuing.", exc)
                continue
            rows += [_row(w) for w in data.get("results") or []]
        if rows:
            new = pd.DataFrame(rows, columns=COLUMNS)
            self.df = (pd.concat([self.df, new], ignore_index=True)
                       .sort_values("fetched_at").drop_duplicates("openalex_id", keep="last"))
            tmp = self.path + ".tmp"
            self.df.to_parquet(tmp, index=False)
            os.replace(tmp, self.path)
```

> In the Streamlit app, a `WorkLookup` writes its cache from multiple sessions. Wrap `_fetch`'s write
> in a module-level `threading.Lock()` and get a single `WorkLookup` via `@st.cache_resource` (plan 05).

---

## 6. Synthetic references (offline runs only): `scholargrid/foundations/synthetic.py`

The synthetic corpus has no OpenAlex data. To exercise the stage in CI, generate deterministic references. Each paper in topic *t* cites 4–8 works from a per-topic pool of 20 (`WSYN{t}{k:02d}`), with Zipf-like weights so a few works are clearly foundational. Two works (`WSYNALL01`, `WSYNALL02`) are cited by every topic; they must lose on lift.

```python
"""Deterministic fake references and works for the synthetic corpus (CI/demo only)."""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np
import pandas as pd

_TOPIC = {"cs.CL": 1, "cs.CV": 2, "cs.LG": 3, "cs.CR": 4, "cs.RO": 5, "cs.DC": 6}


def synthetic_references(df: pd.DataFrame) -> pd.Series:
    out = []
    weights = 1.0 / np.arange(1, 21) ** 1.1
    weights /= weights.sum()
    for aid, cat in zip(df["arxiv_id"].astype(str), df["primary_category"].astype(str)):
        rng = np.random.default_rng(int(hashlib.sha1(aid.encode()).hexdigest()[:8], 16))
        t = _TOPIC.get(cat, 9)
        picks = rng.choice(20, size=int(rng.integers(4, 9)), replace=False, p=weights)
        refs = [f"WSYN{t}{k:02d}" for k in sorted(picks)] + ["WSYNALL01", "WSYNALL02"]
        out.append(" ".join(refs))
    return pd.Series(out, index=df.index)


def synthetic_works(ids: List[str]) -> pd.DataFrame:
    rows = []
    for oid in ids:
        if not oid.startswith("WSYN"):
            continue
        k = int(oid[-2:]) if oid[-2:].isdigit() else 0
        survey = k in (3, 7)
        rows.append({"openalex_id": oid, "title": (f"A Survey of Synthetic Topic {oid[4:-2]}" if survey else
                                                   f"Foundational Work {k} on Topic {oid[4:-2]}"),
                     "year": 2000 + k, "cited_by_count": 5000 // (k + 1), "doi": "", "arxiv_id": "",
                     "type": "review" if survey else "article", "venue": "Synthetic Journal",
                     "abstract": "Synthetic foundational paper used for offline tests.",
                     "authors": "A. Founder", "url": f"https://openalex.org/{oid}", "fetched_at": 0.0})
    return pd.DataFrame(rows)
```

---

## 7. Backbone scoring: `scholargrid/foundations/backbone.py`

```python
"""Area-specific co-citation backbone."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd


def reference_lists(refs: pd.Series) -> List[List[str]]:
    return [str(r).split() if isinstance(r, str) and r else [] for r in refs.tolist()]


def backbone(ref_lists: List[List[str]], labels: np.ndarray, min_citing: int, min_share: float,
             top: int, eps: float) -> Dict[int, pd.DataFrame]:
    """{area_id: DataFrame[openalex_id, freq, share, lift, score]} for every clustered area."""
    has = np.array([bool(r) for r in ref_lists])
    n_all = int(has.sum())
    if n_all == 0:
        return {}
    glob: Counter = Counter()
    for r in ref_lists:
        glob.update(set(r))
    out: Dict[int, pd.DataFrame] = {}
    for cid in sorted({int(c) for c in labels if c != -1}):
        idx = np.where((labels == cid) & has)[0]
        if len(idx) == 0:
            continue
        cnt: Counter = Counter()
        for i in idx:
            cnt.update(set(ref_lists[i]))
        n = len(idx)
        rows = []
        for w, f in cnt.items():
            share = f / n
            if f < min_citing or share < min_share:
                continue
            lift = (share + eps) / (glob[w] / n_all + eps)
            rows.append({"openalex_id": w, "freq": f, "n_citing_base": n, "share": round(share, 4),
                         "lift": round(lift, 3), "score": round(f * float(np.log1p(lift)), 4)})
        if rows:
            out[cid] = pd.DataFrame(rows).sort_values("score", ascending=False).head(top).reset_index(drop=True)
    return out
```

---

## 8. Reading path: `scholargrid/foundations/path.py`

One function serves both offline (per area) and online (per workspace subset).

```python
"""Ordered reading path: surveys -> foundations -> key recent -> frontier."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .surveys import is_survey


def _age_years(year, ref_year: int) -> float:
    try:
        return max(0.0, ref_year - int(year))
    except (TypeError, ValueError):
        return 5.0


def _item(step: str, *, paper_id: str, title: str, year, url: str, cites: int, reason: str,
          in_corpus: bool, bundle_index: Optional[int] = None, survey: bool = False) -> Dict:
    return {"step": step, "paper_id": paper_id, "title": title, "year": int(year) if year else None,
            "url": url, "cited_by_count": int(cites or 0), "reason": reason, "in_corpus": in_corpus,
            "bundle_index": bundle_index, "is_survey": survey}


def reading_path(papers: pd.DataFrame, idx: np.ndarray, embeddings: np.ndarray, works: pd.DataFrame,
                 backbone_rows: pd.DataFrame, problems_by_paper: Dict[str, int], sizes: Dict[str, int],
                 recent_months: int, frontier_months: int, n_base: int) -> List[Dict]:
    """``papers`` = full bundle df; ``idx`` = bundle row indices of the subset (area or workspace);
    ``works`` = WorkLookup rows for backbone ids; ``backbone_rows`` = scored backbone for the subset."""
    sub = papers.iloc[idx]
    ref = pd.Timestamp(papers["date"].max())
    ref_year = int(ref.year)
    seen: set = set()
    out: List[Dict] = []

    def add(item: Dict) -> None:
        key = item["paper_id"]
        if key and key not in seen:
            seen.add(key)
            out.append(item)

    # 1) surveys: in-subset surveys + backbone works that are reviews
    cand = []
    for i, r in zip(idx, sub.itertuples()):
        if is_survey(r.title, r.abstract):
            y = pd.Timestamp(r.date).year
            cand.append((int(r.cited_by_count) / (_age_years(y, ref_year) + 1), _item(
                "survey", paper_id=str(r.arxiv_id), title=r.title, year=y, url=f"https://arxiv.org/abs/{r.arxiv_id}",
                cites=int(r.cited_by_count), reason=f"Survey from {y} covering this topic "
                f"({int(r.cited_by_count)} citations).", in_corpus=True, bundle_index=int(i), survey=True)))
    w_by_id = works.set_index("openalex_id") if len(works) else pd.DataFrame()
    for b in backbone_rows.itertuples() if len(backbone_rows) else []:
        if b.openalex_id in getattr(w_by_id, "index", []):
            w = w_by_id.loc[b.openalex_id]
            if is_survey(w["title"], w["abstract"], w["type"]):
                cand.append((w["cited_by_count"] / (_age_years(w["year"], ref_year) + 1), _item(
                    "survey", paper_id=w["arxiv_id"] or b.openalex_id, title=w["title"], year=w["year"],
                    url=w["url"], cites=w["cited_by_count"], reason=f"Survey cited by {b.freq} papers here.",
                    in_corpus=False, survey=True)))
    for _, it in sorted(cand, key=lambda t: -t[0])[: sizes.get("survey", 3)]:
        add(it)

    # 2) foundations: backbone by score, displayed oldest first
    fnd = []
    for b in backbone_rows.itertuples() if len(backbone_rows) else []:
        if b.openalex_id not in getattr(w_by_id, "index", []):
            continue
        w = w_by_id.loc[b.openalex_id]
        if not w["title"]:
            continue
        fnd.append(_item("foundation", paper_id=w["arxiv_id"] or b.openalex_id, title=w["title"],
                         year=w["year"], url=w["url"], cites=w["cited_by_count"],
                         reason=f"Cited by {b.freq} of the {n_base} papers here, a shared foundation.",
                         in_corpus=bool(w["arxiv_id"] and (papers["arxiv_id"] == w["arxiv_id"]).any()),
                         survey=False))
    for it in sorted(fnd[: sizes.get("foundation", 6)], key=lambda x: x["year"] or 0):
        add(it)

    # 3) key recent: last recent_months, by citation velocity (fallback citations per month)
    recent_cut = ref - pd.DateOffset(months=recent_months)
    rec = sub[pd.to_datetime(sub["date"]) >= recent_cut]
    if len(rec):
        months = ((ref - pd.to_datetime(rec["date"])).dt.days / 30.4).clip(lower=1)
        vel = rec["citation_velocity"].astype(float)
        rate = np.where(vel > 0, vel, rec["cited_by_count"].astype(float) / months)
        for pos in np.argsort(-rate)[: sizes.get("recent", 6)]:
            r = rec.iloc[int(pos)]
            add(_item("recent", paper_id=str(r["arxiv_id"]), title=r["title"], year=pd.Timestamp(r["date"]).year,
                      url=f"https://arxiv.org/abs/{r['arxiv_id']}", cites=int(r["cited_by_count"]),
                      reason=f"One of the most-cited recent papers here ({float(rate[int(pos)]):.1f} citations/month).",
                      in_corpus=True, bundle_index=int(papers.index.get_loc(r.name))))

    # 4) frontier: last frontier_months, central to the subset, prefer stated open problems
    fr_cut = ref - pd.DateOffset(months=frontier_months)
    fmask = pd.to_datetime(sub["date"]).to_numpy() >= np.datetime64(fr_cut)
    if fmask.any():
        e = np.asarray(embeddings[idx], dtype=np.float32)
        centroid = e.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        sims = e @ centroid
        fidx = np.where(fmask)[0]
        bonus = np.array([0.05 if problems_by_paper.get(str(sub.iloc[j]["arxiv_id"]), 0) else 0.0 for j in fidx])
        for j in fidx[np.argsort(-(sims[fidx] + bonus))][: sizes.get("frontier", 5)]:
            r = sub.iloc[int(j)]
            has_p = problems_by_paper.get(str(r["arxiv_id"]), 0)
            add(_item("frontier", paper_id=str(r["arxiv_id"]), title=r["title"], year=pd.Timestamp(r["date"]).year,
                      url=f"https://arxiv.org/abs/{r['arxiv_id']}", cites=int(r["cited_by_count"]),
                      reason=(f"Recent ({pd.Timestamp(r['date']):%b %Y}) and central to this topic"
                              + ("; it lists open problems you can build on." if has_p else ".")),
                      in_corpus=True, bundle_index=int(idx[int(j)])))
    return out
```

> `papers.index.get_loc(r.name)` assumes the bundle DataFrame has a default RangeIndex, which
> `load_papers` produces. Assert this once in the stage.

---

## 9. Optional LLM reasons: `scholargrid/foundations/prompts.py`

Template reasons are honest but dry. With `foundations.llm_reasons: true` and an available LLM, add a plain-language **why read** and **what you'll learn** to each survey and foundation item. That is ~10 items per area, computed offline in a batch and cached.

```python
from pydantic import BaseModel, field_validator
from typing import Literal

from ..llm.fakes import register_fake
from ..llm.prompts import Prompt, register


class ReadingReason(BaseModel):
    why_read: str             # <= 25 words, for a newcomer
    what_you_learn: str       # <= 25 words
    level: Literal["intro", "intermediate", "advanced"]

    @field_validator("why_read", "what_you_learn", mode="before")
    @classmethod
    def _c(cls, v):
        s = " ".join(str(v or "").split())
        return s[:220]


register(Prompt(
    name="reading_reason", version="1",
    system=("You help a beginner researcher decide what to read. Given one paper's title and abstract and "
            "the topic it belongs to, write why a newcomer to the topic should read it and what they will "
            "learn. Base everything on the abstract only. Plain English, no hype, no citations counts."),
    user="Topic: $topic\nReading step: $step\n<abstract>\nTitle: $title\n$abstract\n</abstract>",
    packable=True,
))


@register_fake("reading_reason")
def _fake(v: dict) -> dict:
    return {"why_read": f"Introduces ideas central to {v['topic']}.",
            "what_you_learn": f"The core concepts behind {str(v['title'])[:60]}.", "level": "intro"}
```

Add `"scholargrid.foundations.prompts"` to `FEATURE_PROMPT_MODULES`. Use task `reason` (plan 01 config). Items without an abstract keep the template reason only. If the free quota is used up (`QuotaExhausted`), keep the template reasons; they are filled in on a later run (results are cached).

---

## 10. Stage, bundle files and loading

### 10.1 `scholargrid/foundations/__init__.py` → `build_foundations(cfg, bundle_dir) -> stats`

Steps:
1. `b = load_bundle(cfg, directory=cfg.processed_dir)`.
2. `refs = b.df["references"]`. If `cfg["data"]["source"] == "synthetic"` and all are empty, use `synthetic_references(b.df)`.
3. `bb = backbone(reference_lists(refs), b.labels, …)` with config values.
4. `lookup = WorkLookup(cfg, synthetic=…)` → `works = lookup.get(all backbone ids)`.
5. `problems_by_paper` = count of `b.problems` per `arxiv_id` (empty dict if none).
6. For each area: `reading_path(b.df, np.where(b.labels == cid)[0], b.embeddings, works, bb.get(cid, empty), problems_by_paper, sizes, …, n_base=bb[cid].n_citing_base[0] or area size)`.
7. Optional LLM reasons: collect unique (paper_id, step, topic) for survey/foundation items that have an abstract → `llm.structured_many("reason", "reading_reason", ReadingReason, items)` (packable; `llm.tasks.reason.pack_size` items per request) with `ReadingReason` → attach `why_read`, `what_you_learn`, `level` to items. Catch `LLMError` / `LLMUnavailable` and keep templates.
8. Write `foundations.parquet` (all areas' backbone rows joined with work metadata plus `area_id`) and `reading_paths.json` (`{str(area_id): [items]}`) into `cfg.processed_dir`.
9. Return stats: `areas_with_path`, `mean_items`, `works_looked_up`, `works_missing`, `pre_2020_foundations` (count of foundation items with year < 2020, a key signal that this works).

### 10.2 `scholargrid/pipeline.py`

```python
STAGES = ["ingest", "enrich", "fulltext", "extract", "embed", "analyze", "foundations", "validate", "publish"]
_PROGRESS["foundations"] = (0.8, "Finding foundational papers and reading paths…")


def stage_foundations(ctx: Context) -> None:
    from .foundations import build_foundations

    cfg = ctx.cfg
    with stage_timer("foundations", ctx.timings) as info:
        stats = build_foundations(cfg, cfg.processed_dir) if cfg["foundations"]["enabled"] else {}
        info.update(**{k: v for k, v in stats.items() if isinstance(v, (int, float))})
    meta_path = os.path.join(cfg.processed_dir, "meta.json")
    meta = load_json(meta_path)
    meta["foundations"] = stats
    save_json(meta, meta_path)
    if "meta" in ctx.data:
        ctx.data["meta"]["foundations"] = stats
```

If `foundations.enabled` is false, still write an empty `reading_paths.json` (`{}`) and an empty `foundations.parquet`.

### 10.3 `scholargrid/artifacts.py`

Add `FOUNDATIONS_PARQUET = "foundations.parquet"` and `READING_PATHS_JSON = "reading_paths.json"`. Add Bundle fields `foundations: pd.DataFrame` (default empty) and `reading_paths: Dict` (default `{}`). Load them optionally.

---

## 11. UI: "Read first" panel on the Areas detail page (`app/views/areas.py::_detail`)

Below the existing two-column layout, add:

```python
    path = (bundle.reading_paths or {}).get(str(cid)) or []
    with ui.panel("area-read-first"):
        ui.panel_header("Read first", f"{len(path)} papers in order")
        if not path:
            ui.note("No reading path yet for this area (needs citation data).")
        labels = {"survey": "Start with an overview", "foundation": "Foundations",
                  "recent": "Key recent work", "frontier": "The frontier"}
        for step in ("survey", "foundation", "recent", "frontier"):
            items = [p for p in path if p["step"] == step]
            if not items:
                continue
            st.markdown(f"<div class='sg-section'>{labels[step]}</div>", unsafe_allow_html=True)
            for n, p in enumerate(items):
                ui.paper_card(p["title"], p["paper_id"], url=p["url"], authors="", cites=p["cited_by_count"],
                              date=str(p["year"] or ""), snippet=p.get("why_read") or p["reason"], delay=n)
```

Add a helper to `app/ui.py` for a "not in collection" chip: items with `in_corpus=False` show the chip **"Outside the collection · via OpenAlex"**, so users understand why the map doesn't show them.

---

## 12. Tests: `tests/test_foundations.py`

1. **Survey regex:** true for "A Survey of Retrieval-Augmented Generation", "SoK: Prompt Injection", "Tutorial on …"; false for "Peer Review Generation with LLMs" and "Automated Code Review".
2. **Backbone lift:** a constructed `ref_lists` where `WALL` is cited by every paper in two areas and `WA1` only in area 0 → `WA1` ranks above `WALL` in area 0, and `WALL` has lift ≈ 1.
3. **Thresholds:** works below `min_citing` or `min_share` are excluded.
4. **WorkLookup:** monkeypatch `scholargrid.sources.http_get` to return a fake JSON with two works. Assert the request filter is `ids.openalex:W1|W2`, rows are parsed (abstract rebuilt, arXiv ID found from a location URL), the cache file is written, and a second `get` makes no HTTP call. The deadline test: `deadline=time.time() - 1` → no calls, empty result, no exception.
5. **reading_path:** a small synthetic bundle (use the `app_config`-style fixture that runs the pipeline) → every area with references has `foundation` items sorted oldest first; no duplicate `paper_id`; the `survey` step contains the synthetic review works (`k in (3, 7)`); `WSYNALL01` is not among the top foundations.
6. **Stage end-to-end:** `run(cfg)` then `load_bundle(cfg).reading_paths` is non-empty, and `meta["foundations"]["pre_2020_foundations"] > 0` (synthetic years 2000–2019).
7. **App:** `tests/test_app.py` already renders `areas`. Add a test that opens an area detail (`at.session_state["area_open"] = <first cluster id>`), runs, and asserts the text "Read first" appears.

---

## 13. Acceptance checks

```bash
ruff check . && mypy && pytest -q
python -m scholargrid.pipeline --config configs/ci.yaml --no-publish
python - <<'EOF'
from scholargrid.config import load_config
from scholargrid.artifacts import load_bundle
cfg = load_config("configs/ci.yaml"); b = load_bundle(cfg, directory=cfg.processed_dir)
cid, path = next(iter(b.reading_paths.items()))
for p in path: print(p["step"], p["year"], p["title"][:70], "|", p["reason"])
EOF
streamlit run app/streamlit_app.py      # Areas → open an area → "Read first" panel
```

On a real cs.CL build (`configs/domain_nlp_dev.yaml` with OpenAlex enabled and `OPENALEX_MAILTO` set, built by the `dev-build` workflow from plan 01 §8.2 on GitHub; download the artifact into `data/releases/` and open it with `streamlit run`), check that the foundations of an LLM-related area include well-known pre-2020 papers (e.g. the Transformer paper, BERT) and that the surveys are real surveys. Record 3 example paths in `Docs/runbook.md`.

## 14. Pitfalls

- OpenAlex `references` cover only matched works (≈ 30–45% of recent arXiv papers in the current release). Paths for areas with low coverage are thin; show the coverage ("based on references from 112 of 410 papers").
- Universal citations (Adam, ResNet, the Transformer paper) dominate raw counts. Lift fixes most of that. Don't remove them by a blocklist; they can be legitimate foundations for some areas.
- Keep OpenAlex requests polite (`mailto`, `LIMITER`). The lookup is cached, and online calls respect `online_deadline_s`.
