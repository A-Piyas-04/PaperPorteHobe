# Plan 05: Topic workspace (query-scoped, fine-grained) and the journey navigation

> **Prerequisites:** plans 01–04 are merged. The bundle has facets, problems, profiles, foundations and reading paths.
> **Produces:**
> 1. `scholargrid/workspace/`: turn **one user query** into a *topic workspace*. Retrieve from the collection and live sources, cluster into **8–12 sub-topics** (plus micro-topics), label them with keywords **and** plain-language LLM summaries, compute activity and top facets, build a **topic-specific reading path**, and cache it.
> 2. a **new journey navigation**: Start → Topic → Read first → (Open problems, plan 06) → (Plan, plan 07) → My project (plan 08), with the existing pages under "More",
> 3. pages `start`, `topic` and `reading`.
>
> **Why:** the global 77-area map is too coarse for a beginner. They need the sub-structure *of their own topic*, in words they understand.

---

## 1. Prompt to paste into Cursor

```text
Implement plan file 05_topic_workspace.md. Follow 00_INDEX_READ_FIRST.md conventions. Use the LLM only
through scholargrid.llm (plan 01) and degrade gracefully when it is unavailable. Order: §3 config →
§4 data model → §5 retrieval → §6 sub-topic clustering → §7 labels (keywords, facets, LLM) →
§8 activity + reading path → §9 build_workspace + cache → §10 navigation → §11 pages → §12 tests.
Keep existing pages working; they move under "More". Finish with §13.
```

---

## 2. Flow and time budget

```
query ──► retrieve (collection hybrid top-N  +  one page of live sources)      ~2–10 s
       ──► vectors (bundle embeddings / live-store vectors; embed missing ≤ 50)  ~1 s
       ──► sub-topics: spherical k-means, k by silhouette; merge tiny; micro-topics  <1 s
       ──► labels: c-TF-IDF keywords + top facets (instant), LLM names + summaries (parallel)  ~5–20 s
       ──► activity per sub-topic, overview (1 LLM call), reading path (backbone of the subset)  ~1–6 s
       ──► cache: data/workspaces/<key>.json  (+ session)                          instant next time
```

Target: **first view < 60 s, repeat view < 1 s.** The page shows progress with `st.status` and renders keyword labels first if the LLM is slow or unavailable.

---

## 3. Config (`scholargrid/schema.py` + `configs/config.yaml`)

```python
class Workspace(_Section):
    enabled: bool = True
    max_papers: int = Field(default=500, ge=50, le=3000)
    min_papers: int = Field(default=25, ge=5)
    live: bool = True                       # add one page of arXiv/OpenAlex/S2 results
    embed_missing_live: int = Field(default=50, ge=0)
    subtopic_k_range: List[int] = Field(default_factory=lambda: [4, 12])
    min_subtopic_size: int = Field(default=8, ge=2)
    micro_min_size: int = Field(default=40, ge=10)
    micro_k_range: List[int] = Field(default_factory=lambda: [2, 4])
    min_silhouette_micro: float = 0.05
    label_with_llm: bool = True
    overview_with_llm: bool = True
    representative_per_subtopic: int = Field(default=8, ge=3)
    cache_dir: str = "data/workspaces"
    cache_ttl_hours: float = Field(default=72.0, gt=0)
    max_cached: int = Field(default=500, ge=10)
```

YAML block `workspace:` with the same defaults. Add `paths`-style property `workspace_dir` in `Config`.

---

## 4. Data model: `scholargrid/workspace/model.py`

Plain dataclasses that serialize to and from JSON (they are cached on disk and in session state).

```python
"""Topic workspace data model (JSON-serializable)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class WsPaper:
    paper_id: str                 # arXiv id, or live paper_id
    title: str
    abstract: str
    date: str                     # YYYY-MM-DD or YYYY
    cited_by_count: int
    url: str
    bundle_index: Optional[int]   # None for live papers
    is_new: bool                  # found online, not in the analysed collection
    sim: Optional[float]
    subtopic: int = -1
    micro: int = -1


@dataclass
class Subtopic:
    id: int
    parent: Optional[int]                 # None = top level; else id of the parent sub-topic
    keyword_label: str                    # c-TF-IDF pretty label (always present)
    name: str                             # LLM name, or keyword_label
    summary: str = ""                     # 2 plain-English sentences (LLM) or ""
    key_concepts: List[Dict[str, str]] = field(default_factory=list)   # [{term, meaning}]
    beginner_entry: str = ""
    keywords: List[str] = field(default_factory=list)
    size: int = 0
    n_collection: int = 0
    recent_share: float = 0.0             # share of papers from the last 12 months
    monthly: Dict[str, int] = field(default_factory=dict)
    top_facets: Dict[str, List[List]] = field(default_factory=dict)   # facet -> [[value, count], ...]
    representative: List[str] = field(default_factory=list)          # paper_ids
    n_open_problems: int = 0


@dataclass
class TopicWorkspace:
    key: str
    query: str
    bundle_version: str
    created_at: float
    papers: List[WsPaper]
    subtopics: List[Subtopic]
    overview: Dict = field(default_factory=dict)        # {summary, how_they_relate, prerequisites:[...]}
    reading_path: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)
    llm_used: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_json(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict) -> TopicWorkspace:
        return cls(**{**d, "papers": [WsPaper(**p) for p in d["papers"]],
                      "subtopics": [Subtopic(**s) for s in d["subtopics"]]})

    def top_level(self) -> List[Subtopic]:
        return [s for s in self.subtopics if s.parent is None]

    def children(self, sid: int) -> List[Subtopic]:
        return [s for s in self.subtopics if s.parent == sid]

    def papers_in(self, sid: int) -> List[WsPaper]:
        return [p for p in self.papers if p.subtopic == sid or p.micro == sid]
```

---

## 5. Retrieval: `scholargrid/workspace/retrieve.py`

- **Collection:** `scholargrid.search.search(query, …, top_k=max_papers)` (existing; hybrid dense + BM25 with the related-similarity cutoff). The `top_k` argument caps exact + related together.
- **Live:** if `workspace.live` and `searcher.live_enabled`: `searcher.run(query, pages=1, live=True)`. Keep results with `is_new=True` (not in the collection). Their vectors come from `searcher.store`: `j = store.find_row(paper_id)`, used only if `store.has_vec[j]`. Embed up to `embed_missing_live` missing ones with `bundle.embedder.transform(paper_text)`. Others are kept in the list but excluded from clustering (`subtopic=-1`, shown under "Other papers found online").
- If fewer than `min_papers` collection results: add a warning ("Few papers match; try a broader phrase") and still build.

```python
def retrieve(query: str, bundle, searcher, cfg) -> tuple[list[WsPaper], np.ndarray, list[str]]:
    """Returns (papers, vectors aligned to papers that have one, warnings). Papers without vectors
    are at the end of the list and have no row in `vectors`."""
```

Implementation details:
- Build `WsPaper` from collection results: `bundle_index=r["bundle_index"]` and `url=f"https://arxiv.org/abs/{arxiv_id}"`. Truncate the abstract to 1,200 chars for the cache.
- De-duplicate live results against the collection using `searcher.bundle_keys.find(rec)` (existing `KeyIndex`). `run()` already marks those `is_new=False`.
- Vectors: `bundle.embeddings[idx]` as `float32`, then `l2_normalize`.

---

## 6. Sub-topic clustering: `scholargrid/workspace/subtopics.py`

Spherical k-means on unit vectors is stable for 50–3,000 points, fast, and reproducible with a seed. Choose *k* by cosine silhouette. HDBSCAN on few hundred raw 384-d vectors gives too much noise for this purpose.

```python
"""Cluster a workspace into sub-topics and micro-topics."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)


def choose_k(x: np.ndarray, k_range: Tuple[int, int], seed: int) -> Tuple[int, np.ndarray, float]:
    from sklearn.metrics import silhouette_score

    lo, hi = k_range
    hi = min(hi, max(lo, len(x) // 10))
    best = (lo, _kmeans(x, lo, seed), -1.0)
    for k in range(lo, hi + 1):
        labels = _kmeans(x, k, seed)
        if len(set(labels)) < 2:
            continue
        sample = min(len(x), 2000)
        s = float(silhouette_score(x, labels, metric="cosine", sample_size=sample, random_state=seed))
        if s > best[2]:
            best = (k, labels, s)
    return best


def merge_small(x: np.ndarray, labels: np.ndarray, min_size: int) -> np.ndarray:
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = ids[counts < min_size]
        if len(small) == 0 or len(ids) <= 2:
            return labels
        cents = {c: x[labels == c].mean(axis=0) for c in ids}
        s = int(small[0])
        others = [c for c in ids if c != s]
        target = max(others, key=lambda c: float(cents[s] @ cents[c]))
        labels[labels == s] = target


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(labels, return_counts=True)
    order = {int(c): i for i, c in enumerate(ids[np.argsort(-counts)])}
    return np.array([order[int(c)] for c in labels])


def subtopics(x: np.ndarray, cfg) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Returns (top-level labels 0..K-1, micro labels as global ids >= K or -1, info)."""
    w = cfg["workspace"]
    seed = int(cfg["seed"])
    if len(x) < 2 * int(w["min_subtopic_size"]):
        return np.zeros(len(x), dtype=int), -np.ones(len(x), dtype=int), {"k": 1, "silhouette": None}
    k, labels, sil = choose_k(x, tuple(w["subtopic_k_range"]), seed)
    labels = relabel_by_size(merge_small(x, labels, int(w["min_subtopic_size"])))
    micro = -np.ones(len(x), dtype=int)
    next_id = int(labels.max()) + 1
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < int(w["micro_min_size"]):
            continue
        mk, ml, ms = choose_k(x[idx], tuple(w["micro_k_range"]), seed)
        if ms < float(w["min_silhouette_micro"]) or len(set(ml)) < 2:
            continue
        ml = merge_small(x[idx], ml, max(5, int(w["min_subtopic_size"]) // 2))
        for m in np.unique(ml):
            micro[idx[ml == m]] = next_id
            next_id += 1
    return labels, micro, {"k": int(len(np.unique(labels))), "silhouette": round(sil, 4)}
```

---

## 7. Labels: keywords, facets, LLM names and summaries

### 7.1 Keyword labels (always computed): `labels.py`

Reuse `scholargrid.labeling._c_tf_idf` and `pretty_label`. Build one aggregated document per sub-topic (titles + abstracts), using `CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)`. Remove the query's own words from the candidate keywords, so a "graph neural networks" workspace doesn't label every sub-topic "Graph Neural Networks".

### 7.2 Top facets per sub-topic

From `bundle.facets` (plan 03) for member papers with a `bundle_index`: count canonical `value` per facet type (`task`, `method` with role ≠ baseline, `dataset`, `metric`, `domain`, `language`). Keep the top 5 each as `[[value, count], …]`. Count member problems from `bundle.problems` → `n_open_problems`.

### 7.3 LLM names and summaries: `scholargrid/workspace/prompts.py`

```python
from typing import List

from pydantic import BaseModel, field_validator

from ..llm.fakes import register_fake
from ..llm.prompts import Prompt, register


class Concept(BaseModel):
    term: str
    meaning: str          # one short sentence a newcomer understands


class SubtopicLabel(BaseModel):
    name: str             # <= 6 words, specific, no buzzwords
    summary: str          # 2 sentences: what problem this group works on and how
    key_concepts: List[Concept]
    beginner_entry: str   # one sentence: what to read or try first, based only on the listed papers

    @field_validator("name", mode="before")
    @classmethod
    def _n(cls, v):
        return " ".join(str(v or "").split()[:8])

    @field_validator("key_concepts", mode="before")
    @classmethod
    def _k(cls, v):
        return list(v or [])[:6]


class TopicOverview(BaseModel):
    summary: str                  # 3-4 sentences about the whole topic as reflected by these papers
    how_they_relate: str          # 2 sentences on how the sub-topics relate
    prerequisites: List[str]      # 3-6 things to know first (concepts/skills)


register(Prompt(
    name="subtopic_label", version="1",
    system=("You name and explain one group of related research papers for a beginner researcher. "
            "Use only the evidence given (keywords, frequent methods/datasets, paper titles and one-line "
            "summaries). Be specific: prefer 'Low-rank adapters for LLM fine-tuning' over 'Efficient methods'. "
            "Explain jargon in key_concepts. Do not claim trends or quality."),
    user=("User's topic: $query\nKeywords: $keywords\nFrequent facets: $facets\n"
          "<evidence>\n$papers\n</evidence>"),
    packable=True,
))
register(Prompt(
    name="topic_overview", version="1",
    system=("You orient a beginner researcher in a topic using only the sub-topic summaries provided. "
            "Plain English. No hype, no claims about what is best."),
    user="User's topic: $query\n<evidence>\n$subtopics\n</evidence>",
))


@register_fake("subtopic_label")
def _fake_label(v: dict) -> dict:
    kw = str(v["keywords"]).split(", ")
    return {"name": " ".join(kw[:3]).title() or "Group", "summary": f"Papers about {', '.join(kw[:3])}.",
            "key_concepts": [{"term": k, "meaning": f"A concept related to {k}."} for k in kw[:3]],
            "beginner_entry": "Start with the most cited paper listed."}


@register_fake("topic_overview")
def _fake_overview(v: dict) -> dict:
    return {"summary": f"An overview of {v['query']}.", "how_they_relate": "The groups share methods.",
            "prerequisites": ["linear algebra", "machine learning basics"]}
```

- The evidence for a sub-topic is up to `representative_per_subtopic` papers closest to its centroid. Each line is `- {title} :: {one_line_summary from bundle.profiles, else first abstract sentence}`, wrapped with `wrap_untrusted("evidence", …)`.
- Label all sub-topics with **packed** calls: `llm.structured_many("label", "subtopic_label", SubtopicLabel, [(str(sid), vars), ...])` (register the prompt with `packable=True`; `llm.tasks.label.pack_size` = 6, so 12 sub-topics = 2 requests). Then make one `llm.structured("label", "topic_overview", TopicOverview, …)` over the generated summaries. A whole workspace costs **≈ 3 requests** of free quota, and repeat views cost 0 (cache).
- On `LLMError` / `LLMUnavailable` / `QuotaExhausted` per item: keep `name = keyword_label`, summary empty, and add a workspace warning ("Summaries unavailable right now (free AI quota used up); showing keyword labels" for quota). Don't cache a workspace whose labels failed only because of quota: mark `ws.stats["labels_pending"] = True`, so the next visit after the reset retries the labels.
- Register the module in `FEATURE_PROMPT_MODULES`.
- Add `ui.get_llm()` to `app/ui.py` now: `@st.cache_resource` wrapper returning `scholargrid.llm.get_llm(get_config())`. Every page passes this client into library calls. Plan 11 extends it with the online budget guard and bring-your-own-key.

---

## 8. Activity and topic reading path

### 8.1 Activity (`activity.py`)
- `monthly`: counts per `YYYY-MM` for papers with full dates.
- `recent_share`: share of papers dated within 12 months of the newest workspace paper.
- Display rule (UI): show counts always. Call a sub-topic **"Recently active"** only if `size ≥ 15` and `recent_share ≥ 0.5`, and say "based on the papers retrieved, not a growth estimate". The rigorous growth statistics stay on the Trends page.

### 8.2 Reading path for the workspace
Reuse plan 04 directly:

```python
from ..foundations.backbone import backbone, reference_lists
from ..foundations.path import reading_path

idx = np.array([p.bundle_index for p in ws.papers if p.bundle_index is not None])
labels = -np.ones(len(bundle.df), dtype=int)
labels[idx] = 0
bb = backbone(reference_lists(bundle.df["references"]), labels, min_citing=max(2, len(idx) // 50),
              min_share=0.02, top=30, eps=float(cfg["foundations"]["lift_epsilon"]))
works = lookup.get(bb[0]["openalex_id"].tolist(), deadline=time.time() + cfg["foundations"]["online_deadline_s"]) \
        if 0 in bb else pd.DataFrame()
ws.reading_path = reading_path(bundle.df, idx, bundle.embeddings, works, bb.get(0, pd.DataFrame()),
                               problems_by_paper, cfg["foundations"]["path_sizes"],
                               cfg["foundations"]["recent_months"], cfg["foundations"]["frontier_months"],
                               n_base=int(len(idx)))
```

`backbone()` still uses global frequencies for lift, so area-agnostic universal citations stay down-weighted. Use a process-wide `WorkLookup` (`ui.get_work_lookup()` with `@st.cache_resource`). In synthetic mode, construct it with `synthetic=True`. Optionally fill `why_read` with the `reading_reason` prompt for survey/foundation items (≤ 9 calls, concurrent, cached).

---

## 9. `build_workspace` + cache: `scholargrid/workspace/build.py`

```python
"""build_workspace(query, bundle, searcher, cfg, progress) -> TopicWorkspace (cached on disk)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Callable, Optional

from ..config import Config
from ..utils import ensure_dir, get_logger
from .model import TopicWorkspace

log = get_logger("workspace")
Progress = Optional[Callable[[str, float], None]]


def workspace_key(query: str, bundle_version: str, cfg: Config) -> str:
    q = " ".join(query.lower().split())
    w = json.dumps(cfg["workspace"], sort_keys=True)
    return hashlib.sha1(f"{bundle_version}|{q}|{w}|{cfg['llm']['provider']}".encode()).hexdigest()[:20]


def load_cached(key: str, cfg: Config) -> Optional[TopicWorkspace]:
    path = os.path.join(cfg.workspace_dir, f"{key}.json")
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > float(cfg["workspace"]["cache_ttl_hours"]) * 3600:
        return None
    with open(path, encoding="utf-8") as fh:
        return TopicWorkspace.from_json(json.load(fh))


def save_cached(ws: TopicWorkspace, cfg: Config) -> None:
    d = ensure_dir(cfg.workspace_dir)
    tmp = os.path.join(d, f"{ws.key}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ws.to_json(), fh)
    os.replace(tmp, os.path.join(d, f"{ws.key}.json"))
    files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")), key=os.path.getmtime)
    for f in files[: max(0, len(files) - int(cfg["workspace"]["max_cached"]))]:
        os.remove(f)


def build_workspace(query: str, bundle, searcher, cfg: Config, *, bundle_version: str,
                    progress: Progress = None, lookup=None, use_cache: bool = True,
                    llm=None) -> TopicWorkspace:
    """Steps: retrieve -> vectors -> subtopics -> labels -> activity -> overview -> reading path -> cache.
    ``llm`` defaults to ``scholargrid.llm.get_llm(cfg)``; the app passes ``ui.get_llm()`` so the online
    budget guard and a user's own key (plan 11) apply. Never call get_llm() deeper inside the steps -
    pass this client down."""
    ...
```

Write each step as a small function in its own module (`retrieve.py`, `subtopics.py`, `labels.py`, `activity.py`, `path.py`). Call `progress(msg, fraction)` between steps, for example "Finding papers…" (0.1), "Grouping into sub-topics…" (0.4), "Writing plain-language summaries…" (0.6), "Building your reading path…" (0.85). Record per-step timings in `ws.stats["timings"]`, plus `n_collection`, `n_live`, `n_live_clustered`, `k`, `silhouette`, and `reference_coverage` (share of collection papers with references).

---

## 10. Navigation: the journey

### 10.1 `app/streamlit_app.py::_build_pages`

```python
    from views import about, areas, explore, home, leads, library, reading, start, topic, trends

    return {
        "start": st.Page(start.render, title="Start", icon=":material/flag:", url_path="start", default=True),
        "topic": st.Page(topic.render, title="Topic", icon=":material/account_tree:", url_path="topic"),
        "reading": st.Page(reading.render, title="Read first", icon=":material/menu_book:", url_path="read-first"),
        # plan 06 adds "gaps", plan 07 "plan", plan 08 replaces "library" with "project"
        "library": st.Page(library.render, title="Reading list", icon=":material/bookmarks:", url_path="reading-list"),
        "home": st.Page(home.render, title="Search papers", icon=":material/search:", url_path="search"),
        "explore": st.Page(explore.render, title="Map", icon=":material/hub:", url_path="explore"),
        "areas": st.Page(areas.render, title="All areas", icon=":material/category:", url_path="areas"),
        "trends": st.Page(trends.render, title="Trends", icon=":material/trending_up:", url_path="trends"),
        "leads": st.Page(leads.render, title="Leads (legacy)", icon=":material/lightbulb:", url_path="leads"),
        "about": st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
    }
```

`home` stays reachable at `/search`. The old `/home` URL path changes; add a one-line note to the README changelog.

### 10.2 `app/ui.py::render_top_nav`

```python
JOURNEY = ["start", "topic", "reading", "gaps", "plan", "project"]
MORE = ["home", "explore", "areas", "trends", "library", "leads", "about"]


def render_top_nav(pages: dict) -> None:
    with st.container(key="topnav"):
        st.markdown("<div class='sg-brand'>ScholarGrid <span class='sg-brand-sub'>From a topic to a "
                    "research question.</span></div>", unsafe_allow_html=True)
        steps = [k for k in JOURNEY if k in pages]
        cols = st.columns([1] * len(steps) + [0.8], gap="small")
        for col, key in zip(cols, steps):
            col.page_link(pages[key])
        with cols[-1], st.popover("More", use_container_width=True):
            for key in MORE:
                if key in pages:
                    st.page_link(pages[key])
```

Add a **journey breadcrumb** helper `ui.journey_bar(current: str)`: a thin row "1 Topic · 2 Read first · 3 Open problems · 4 Plan · 5 Project", with the current step highlighted and completed steps ticked. Completion is read from `st.session_state` flags `j_topic`, `j_read`, `j_gaps`, `j_plan`, `j_project` that each page sets.

### 10.3 Session keys (document them at the top of `app/ui.py`)

| Key | Set by | Meaning |
|---|---|---|
| `topic_query` | start/topic | the active topic text |
| `ws_key` | topic | cache key of the active workspace |
| `ws_selected` | topic | selected sub-topic id |
| `j_*` | each page | journey completion flags |

---

## 11. Pages

### 11.1 `app/views/start.py`

- Hero: eyebrow "Start your research", title "What do you want to research?", lead "Describe a topic, a problem, or a method. We'll map its sub-topics, tell you what to read first, and surface open problems backed by papers."
- `st.form("topic-form")` with `text_input(key="topic_input", max_chars=ui.max_query_chars(), placeholder="e.g. hallucination in retrieval-augmented generation")` and a primary button "Explore this topic". On submit: `st.session_state.topic_query = text`, then `ui.goto("topic")`.
- Example chips: three topic examples. Put them in a new `ui.TOPIC_EXAMPLES` list that fits the current scope; take them from `cfg["scope"]` if it names categories, else general CS.
- "How it works" journey strip: 5 steps with one sentence each.
- "Recent topics" (this session): up to 5 previous queries as buttons.
- Honest footer line: "Covers {n:,} analysed papers ({scope name}) plus live arXiv/OpenAlex/Semantic Scholar results. Open problems come from what authors wrote in their papers."

### 11.2 `app/views/topic.py`

```python
def render() -> None:
    q = st.session_state.get("topic_query", "").strip()
    if not q:
        ui.note("<b>No topic yet.</b> Start by describing what you want to research.")
        if st.button("Choose a topic", type="primary"):
            ui.goto("start")
        return
    ws = ui.get_workspace(q)          # builds with st.status on a cache miss (see below)
    ...
```

`ui.get_workspace(q)`:
1. `key = workspace_key(q, bundle_version(), cfg)`.
2. Session hit (`st.session_state["_ws"][key]`) → return it.
3. Disk hit → `load_cached` → store in the session → return.
4. Otherwise build: `with st.status("Mapping your topic…", expanded=True) as status:` and a progress callback that writes `status.update(label=msg)` plus `st.progress(frac)`. Call `build_workspace(q, get_bundle(), get_searcher(v), cfg, bundle_version=v, progress=cb, lookup=get_work_lookup(), llm=get_llm())`, then `status.update(label="Done", state="complete", expanded=False)`.
5. Catch exceptions → `capture_exception` and a friendly note with a retry button.

Page layout:
1. **Header:** "Topic: {q}", with stats chips: "{n} papers · {k} sub-topics · {n_live} found online".
2. **Overview panel** (if `ws.overview`): summary, "How the parts relate", and "Know these first" (the prerequisites as pills).
3. **Sub-topic grid** (3 columns, sorted by size). Each card shows the name; summary (or keyword pills); "{size} papers · {recent_share:.0%} from the last 12 months"; top methods and datasets as pills; "{n_open_problems} open problems stated by authors"; and an "Open" button that sets `ws_selected`.
4. **Selected sub-topic detail** (when `ws_selected` is set):
   - name, summary, key concepts in a two-column definition list, beginner entry,
   - micro-topics (if any) as tabs with their papers,
   - representative papers using `ui.paper_card` + "Save to reading list" (`views.library.toggle`),
   - buttons: "Show on the map" (existing `explore_hl` mechanism with the sub-topic's bundle indices → `goto("explore")`) and "See open problems here" (plan 06; hide until the page exists).
5. **"Other papers found online"** (live papers without vectors), collapsed in an expander.
6. **Warnings** (few results, LLM unavailable) as `ui.note`.
7. Set `st.session_state.j_topic = True`.

Accessibility: every card button has a unique key (`f"ws_open_{sid}"`). Don't rely on colour alone.

### 11.3 `app/views/reading.py`

- Needs a workspace (same guard as Topic).
- Panel per step, in order, with friendly headings: **"1 · Get the big picture"** (surveys), **"2 · Foundations"**, **"3 · Key recent work"**, **"4 · The frontier"**. Each item is a `ui.paper_card` with `snippet = why_read or reason` and a "what you'll learn" line if present. Items with `in_corpus=False` get the "Outside the collection · via OpenAlex" chip. Each has a "Save to reading list" button.
- Top-right: export the reading path as **Markdown** (ordered list with reasons and links) and **BibTeX** (reuse `views.home._bibtex`).
- A coverage note: "Foundations are based on the references of {k} of {n} papers in this topic (citation data from OpenAlex)."
- If the path is empty: explain why (no citation data in this build) and fall back to the top 10 most-cited workspace papers, labelled as such.
- Set `j_read = True`.

---

## 12. Tests

### 12.1 `tests/test_workspace.py`
1. `choose_k` on three well-separated synthetic blobs (random unit vectors around 3 directions) picks k=3. `merge_small` folds a 2-point cluster into its nearest neighbour.
2. `subtopics()` produces micro-topics only when silhouette ≥ the threshold, and micro ids don't collide with top-level ids.
3. Keyword labels exclude query words (query "graph neural networks" → no label contains "Graph").
4. `build_workspace` on the synthetic bundle (live off, FakeLLM): ≥ 2 sub-topics, all papers with a `bundle_index` have `subtopic ≥ 0`, names come from the fake handler, the overview is present, the reading path is non-empty, and `ws.stats["timings"]` has every step.
5. Cache: a second `build_workspace` returns the same `key`, loading from disk without calling the LLM (count `FakeLLM.calls`). Expired TTL → rebuild.
6. LLM unavailable (`provider: none`): names fall back to keyword labels and the warning is present.
7. JSON round-trip: `TopicWorkspace.from_json(ws.to_json()) == ws` (dataclass equality).

### 12.2 `tests/test_app.py`
- Add `"start"`, `"topic"` and `"reading"` to `PAGES`.
- A topic flow test: `at.session_state["topic_query"] = "language models"`; run `topic` → no exception, and the markdown contains "sub-topics". Then run `reading` with the same session state → contains "Foundations" or the fallback text.
- Navigation test: `streamlit_app.py` renders with `default=True` on `start`.

---

## 13. Acceptance checks

```bash
ruff check . && mypy && pytest -q
python -m scholargrid.pipeline --config configs/ci.yaml --no-publish
streamlit run app/streamlit_app.py
#  1. Start → type "language models" → Topic page shows sub-topics with names/summaries
#  2. Open a sub-topic → concepts, papers, "Show on the map" works
#  3. Read first → 4 steps, export Markdown downloads a file
#  4. Reload the page → instant (disk cache)
#  5. More → Search papers / Map / All areas / Trends still work
```

With a real build and key, time three real topics (`ws.stats["timings"]`) and record them in `Docs/runbook.md`. The target is a first view under 60 s.

## 14. Pitfalls

- Live papers have no `bundle_index`; never index `bundle.df` with them.
- `KMeans` on fewer than `2 * min_subtopic_size` points: return one sub-topic (handled).
- Streamlit reruns the script on every interaction. Never build the workspace inside a widget callback, and always check the session and disk cache first.
- Do not show "growing" labels from the workspace's raw counts. Retrieval is biased toward the query, so only `recent_share` with counts is shown.
- The workspace cache key includes the bundle version: a new release invalidates old workspaces automatically.
