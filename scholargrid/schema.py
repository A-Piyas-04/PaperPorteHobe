"""Pydantic schema for ``configs/*.yaml``.

Validation runs on every ``load_config`` call so a misspelt key or a wrong type
fails immediately with a readable message instead of deep inside a stage.
Unknown keys are rejected inside known sections.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

LiveSource = Literal["arxiv", "openalex", "semantic_scholar"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paths(_Section):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    reports_dir: str = "reports"
    stages_dir: str = "data/stages"
    releases_dir: str = "data/releases"
    live_dir: str = "data/live"


class OaiPmh(_Section):
    endpoint: str = "https://oaipmh.arxiv.org/oai"
    set: str = "cs"
    metadata_prefix: str = "arXiv"
    sleep_seconds: float = 5.0
    max_retries: int = 5


class Data(_Section):
    source: Literal["arxiv_api", "kaggle", "oai_pmh", "synthetic"] = "arxiv_api"
    profile: Literal["dev", "standard", "full", "custom"] = "custom"
    max_papers: Optional[int] = Field(default=1500, ge=1)
    date_start: str = "2023-01-01"
    date_end: str = "auto"
    categories: Union[Literal["all_cs"], List[str]] = "all_cs"
    categories_prefix: str = "cs."
    allow_synthetic_fallback: bool = False
    sampling: Literal["stratified", "newest", "none"] = "stratified"
    use_cache: bool = True
    min_abstract_chars: int = Field(default=200, ge=0)
    language_filter: bool = True
    arxiv_queries: List[str] = Field(default_factory=list)
    arxiv_slice: Literal["month", "none"] = "month"
    arxiv_page_size: int = Field(default=1000, ge=1, le=2000)
    kaggle_json: str = "data/raw/arxiv-metadata-oai-snapshot.json"
    oai_pmh: OaiPmh = Field(default_factory=OaiPmh)

    @field_validator("date_start", "date_end")
    @classmethod
    def _date(cls, v: str) -> str:
        if v == "auto":
            return v
        import datetime as _dt

        _dt.date.fromisoformat(str(v))
        return str(v)


class Storage(_Section):
    backend: Literal["parquet", "duckdb"] = "parquet"
    path: str = "data/scholargrid.duckdb"


class Quality(_Section):
    fail_on_error: bool = False
    min_span_months: int = Field(default=0, ge=0)
    min_papers_per_month: int = Field(default=0, ge=0)
    max_category_share: float = Field(default=0.6, gt=0, le=1)
    max_missing_rate: float = Field(default=0.005, ge=0, le=1)
    max_row_change: float = Field(default=0.2, ge=0)


class RefreshSchedule(_Section):
    lt_3m_days: int = 7
    lt_24m_days: int = 30
    older_days: int = 90


class Enrich(_Section):
    openalex: bool = True
    mailto: str = ""
    batch: int = Field(default=50, ge=1, le=100)
    title_fallback: bool = False
    title_fallback_limit: int = 200
    miss_retry_days: int = 1
    refresh_schedule: RefreshSchedule = Field(default_factory=RefreshSchedule)
    semantic_scholar: bool = False


class Graph(_Section):
    k_neighbors: int = Field(default=6, ge=1)
    coupling_boost: float = 0.15


class Embedding(_Section):
    backend: Literal["auto", "minilm", "tfidf"] = "auto"
    model_name: str = "all-MiniLM-L6-v2"
    model_revision: Optional[str] = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float16"] = "float32"
    incremental: bool = True
    tfidf_dims: int = 256
    tfidf_max_features: int = 20000
    batch_size: int = 64
    query_prefix: str = ""
    max_seq_length: Optional[int] = Field(default=None, ge=16)


class Index(_Section):
    backend: Literal["auto", "exact", "faiss_hnsw"] = "auto"
    exact_threshold: int = 20000
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 128
    recall_sample: int = 200


class Reduce(_Section):
    backend: Literal["auto", "umap", "pca"] = "auto"
    n_components: int = 8
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    fit_sample: int = 60000


class Cluster(_Section):
    backend: Literal["auto", "hdbscan", "dbscan", "kmeans"] = "auto"
    min_cluster_size: Union[int, Literal["auto"]] = 15
    min_cluster_fraction: float = 0.001
    min_samples: int = 5
    hdbscan_cluster_selection_method: Literal["leaf", "eom"] = "leaf"
    fallback_min_clusters: int = 4
    kmeans_k_range: List[int] = Field(default_factory=lambda: [5, 14])
    noise_percentile: float = 98
    stability_seeds: List[int] = Field(default_factory=lambda: [0, 1, 2])
    match_previous: bool = True
    match_min_score: float = 0.3


class Labeling(_Section):
    top_keywords: int = 8
    n_representative: int = 5


class Landscape(_Section):
    backend: Literal["auto", "umap", "pca"] = "auto"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    fit_sample: int = 60000


class Growth(_Section):
    reference_date: str = "auto"
    windows_months: List[int] = Field(default_factory=lambda: [3, 6, 12])
    default_window: int = 6
    min_papers_per_period: int = 30
    bootstrap_samples: int = 500
    ci_level: float = Field(default=0.9, gt=0, lt=1)
    require_stable: bool = True
    max_cv: float = 0.5
    backtest_months: int = 12


class Search(_Section):
    top_k: int = 25
    hybrid: bool = True
    rrf_k: int = 60
    candidate_k: int = 100
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_query_chars: int = 300
    top_k_exact: int = Field(default=100, ge=1)
    top_k_related: int = Field(default=50, ge=0)
    related_min_sim: Optional[float] = None
    related_min_z: float = 2.0
    related_rel_cutoff: float = Field(default=0.5, ge=0, le=1)
    proximity_window: int = Field(default=8, ge=2)
    synonyms: Dict[str, List[str]] = Field(default_factory=dict)
    live: bool = True
    live_sources: List[LiveSource] = Field(
        default_factory=lambda: list(get_args(LiveSource)))
    live_per_source: int = Field(default=200, ge=1)
    live_deadline_s: float = Field(default=10.0, gt=0)
    s2_budget_s: float = Field(default=6.0, gt=0)
    live_cache_hours: float = Field(default=24.0, ge=0)
    live_embed_max: int = Field(default=300, ge=0)
    live_embed_budget_s: float = Field(default=4.0, ge=0)
    cooldown_s: float = Field(default=60.0, ge=0)


class Sparse(_Section):
    enabled: bool = True
    min_corpus_size: int = 1000
    grid_size: int = 24
    low_density_percentile: float = 20
    min_neighbor_density: int = 1
    hd_neighbors: int = 15
    robustness_seeds: List[int] = Field(default_factory=lambda: [0, 1, 2])
    min_robustness: float = 0.5
    max_leads: int = 12


class Validation(_Section):
    queries_file: str = "configs/validation_queries.json"
    judgments_file: Optional[str] = None
    search_top_k: int = 10
    recall_k: int = 25
    alternate_cutoffs_months: List[int] = Field(default_factory=lambda: [3, 6])
    enforce_gates: bool = False
    gates: Dict[str, float] = Field(default_factory=dict)
    required_gates: List[str] = Field(default_factory=list)


class Release(_Section):
    enabled: bool = True
    keep_last: int = Field(default=5, ge=1)
    pin: Optional[str] = None
    write_csv: bool = False


class Monitoring(_Section):
    json_logs: bool = False
    stale_after_days: int = 14


class App(_Section):
    max_graph_nodes: int = Field(default=20000, ge=100)
    max_area_papers: int = Field(default=400, ge=10)
    pre_warm: bool = True
    allow_grow: bool = True
    show_legacy_leads: bool = False      # the old "Investigation leads" page (replaced by Open problems)


LlmMode = Literal["live", "fake", "none"]          # live = call the configured providers
Thinking = Literal["default", "minimal", "low", "medium", "high"]


class LlmProvider(_Section):
    kind: Literal["gemini", "openai_compat"]
    enabled: bool = True
    api_key_env: str = ""                       # name of the env var holding the key
    base_url: Optional[str] = None              # openai_compat only
    json_mode: Literal["schema", "object"] = "schema"   # openai_compat: json_schema supported, or json_object only
    quota_tz: str = "UTC"                       # timezone in which the daily quota resets


class LlmLimits(_Section):
    """Copy from the provider console (AI Studio rate-limit page, Groq console, ...)."""
    rpm: Optional[int] = Field(default=None, ge=1)
    rpd: Optional[int] = Field(default=None, ge=1)
    tpm: Optional[int] = Field(default=None, ge=1)


class LlmTask(_Section):
    route: List[str]                            # ordered "provider:model" entries; first available wins
    thinking: Thinking = "default"
    max_output_tokens: int = Field(default=8192, ge=256, le=65536)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    pack_size: int = Field(default=1, ge=1, le=20)        # items per request for structured_many()
    expected_output_tokens: int = Field(default=1500, ge=1)   # per item; for estimates


class LlmPrice(_Section):
    input: float = 0.0                          # USD / 1M tokens (0 = free tier)
    output: float = 0.0


class LlmBudget(_Section):
    offline_max_requests: Optional[int] = Field(default=None, ge=1)   # per pipeline run; None = until quota
    online_daily_requests: int = Field(default=300, ge=0)             # enforced in plan 11
    online_session_requests: int = Field(default=40, ge=0)            # enforced in plan 11


def _t(route: List[str], thinking: str = "default", max_out: int = 8192, pack: int = 1,
       expected: int = 1500) -> LlmTask:
    return LlmTask(route=route, thinking=thinking, max_output_tokens=max_out, pack_size=pack,  # type: ignore[arg-type]
                   expected_output_tokens=expected)


LITE, FLASH = "gemini:gemini-3.5-flash-lite", "gemini:gemini-3.8-flash"


class Llm(_Section):
    provider: LlmMode = "none"
    providers: Dict[str, LlmProvider] = Field(default_factory=lambda: {
        "gemini": LlmProvider(kind="gemini", api_key_env="GEMINI_API_KEY", quota_tz="America/Los_Angeles"),
    })
    limits: Dict[str, LlmLimits] = Field(default_factory=dict)       # key: "provider:model"
    tasks: Dict[str, LlmTask] = Field(default_factory=lambda: {
        "extract": _t([LITE], "minimal", 16384, pack=4, expected=1200),
        "label": _t([LITE], "minimal", 12288, pack=6, expected=400),
        "reason": _t([LITE], "minimal", 4096, pack=6, expected=200),
        "judge": _t([FLASH, LITE], "low", 12288, pack=5, expected=400),
        "synthesize_gap": _t([FLASH, LITE], "medium", 16384, expected=1500),
        "plan": _t([FLASH, LITE], "medium", 24576, expected=3500),
        "check": _t([LITE], "minimal", 1024, expected=50),
    })
    prices: Dict[str, LlmPrice] = Field(default_factory=dict)        # only if you enable billing somewhere
    max_concurrency: int = Field(default=2, ge=1, le=16)
    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=8)
    max_wait_s: float = Field(default=90.0, ge=0)    # longest wait for a per-minute slot before trying the next route
    cache_dir: str = "data/llm_cache"
    usage_dir: str = "data/llm_usage"
    budget: LlmBudget = Field(default_factory=LlmBudget)
    allow_byok: bool = True

    @model_validator(mode="after")
    def _routes_known(self) -> Llm:
        for name, task in self.tasks.items():
            for entry in task.route:
                provider, sep, model = entry.partition(":")
                if not sep or not model:
                    raise ValueError(f"llm.tasks.{name}.route entry {entry!r} must be 'provider:model'")
                if provider not in self.providers:
                    raise ValueError(f"llm.tasks.{name}.route entry {entry!r}: unknown provider {provider!r} "
                                     f"(known: {sorted(self.providers)})")
        return self


class Scope(_Section):
    """Which papers get the expensive treatment (full text + LLM extraction)."""
    name: str = "all"
    categories: Union[Literal["all"], List[str]] = "all"
    min_date: Optional[str] = None
    max_papers: Optional[int] = Field(default=None, ge=1)
    priority: Literal["most_cited", "newest", "random"] = "most_cited"


SectionKindName = Literal[
    "abstract", "introduction", "background", "related_work", "method", "experiments", "results",
    "discussion", "limitations", "future_work", "threats_to_validity", "broader_impact", "ethics",
    "conclusion", "appendix", "acknowledgements", "references", "other"]
FulltextSource = Literal["arxiv_html", "ar5iv", "pdf"]
_FULLTEXT_ORDER: List[FulltextSource] = ["arxiv_html", "ar5iv", "pdf"]
_KEEP_KINDS: List[SectionKindName] = [
    "abstract", "introduction", "experiments", "results", "discussion", "limitations",
    "future_work", "threats_to_validity", "broader_impact", "conclusion"]


class Fulltext(_Section):
    enabled: bool = True
    source: Literal["arxiv", "synthetic", "none"] = "arxiv"   # synthetic: generated, offline (CI/demo)
    order: List[FulltextSource] = Field(default_factory=lambda: list(_FULLTEXT_ORDER))
    pdf_backend: Literal["pypdf"] = "pypdf"                   # pymupdf (AGPL) is deferred: LATER_NOT_NOW.md
    request_interval_s: float = Field(default=3.0, ge=3.0)    # arXiv: max 1 request / 3 s
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=6)
    max_download_mb: float = Field(default=30.0, gt=0)
    store_dir: str = "data/fulltext"
    keep_kinds: List[SectionKindName] = Field(default_factory=lambda: list(_KEEP_KINDS))
    max_section_chars: int = Field(default=8000, ge=500)
    max_paper_chars: int = Field(default=40000, ge=2000)
    refetch_failed_days: int = Field(default=30, ge=1)
    flush_every: int = Field(default=50, ge=1)
    max_papers_per_run: Optional[int] = Field(default=None, ge=1)   # spread a big backfill over several CI runs


class Extract(_Section):
    enabled: bool = True
    mode: Literal["auto", "llm", "heuristic"] = "auto"
    store_dir: str = "data/extract"
    max_input_chars: int = Field(default=24000, ge=2000)
    section_budget: Dict[str, int] = Field(default_factory=lambda: {
        "abstract": 3000, "introduction": 3500, "experiments": 3500, "results": 2500, "discussion": 3000,
        "limitations": 4000, "future_work": 3000, "threats_to_validity": 2000, "conclusion": 3000,
        "broader_impact": 1500})
    max_cue_sentences: int = Field(default=25, ge=0)
    abstract_only_fallback: bool = True        # papers without full text: extract from title + abstract
    max_llm_attempts: int = Field(default=3, ge=1)   # stop retrying a paper the LLM keeps failing on
    quote_min_ratio: int = Field(default=90, ge=50, le=100)
    max_problems_per_paper: int = Field(default=6, ge=0)
    max_findings_per_paper: int = Field(default=5, ge=0)
    on_quota: Literal["heuristic", "skip", "stop"] = "heuristic"   # when the free quota runs out mid-run
    canonical_min_cosine: float = Field(default=0.88, gt=0, le=1)
    canonical_min_string_ratio: int = Field(default=80, ge=0, le=100)


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


GapType = Literal["stated_open_problem", "long_standing_problem", "unexplored_combination", "evaluation_gap",
                  "unsettled_question", "missing_bridge"]
GapDetector = Literal["stated", "combinatorial", "evaluation", "unsettled", "bridge"]
_GAP_DETECTORS: List[GapDetector] = ["stated", "combinatorial", "evaluation", "unsettled", "bridge"]


class Gaps(_Section):
    enabled: bool = True
    detectors: List[GapDetector] = Field(default_factory=lambda: list(_GAP_DETECTORS))
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

    @field_validator("combo_pairs")
    @classmethod
    def _pairs(cls, v: List[str]) -> List[str]:
        ok = {"method", "task", "domain", "dataset", "language", "modality"}
        for p in v:
            a, sep, b = p.partition(":")
            if not sep or a not in ok or b not in ok:
                raise ValueError(f"gaps.combo_pairs entry {p!r} must be '<facet>:<facet>' from {sorted(ok)}")
        return v


class Plan(_Section):
    enabled: bool = True
    max_references: int = Field(default=16, ge=4, le=40)
    related_top: int = Field(default=12, ge=0)          # extra related papers by similarity to the statement
    foundations_top: int = Field(default=4, ge=0)
    weeks_options: List[int] = Field(default_factory=lambda: [6, 8, 10, 12])
    default_weeks: int = 10
    max_background_chars: int = Field(default=400, ge=0)
    people_top: int = Field(default=8, ge=0)
    venues_top: int = Field(default=6, ge=0)


class Persistence(_Section):
    enabled: bool = True
    db_url_env: str = "SCHOLARGRID_DB_URL"
    default_url: str = "sqlite:///data/app.db"
    auth: Literal["none", "oidc"] = "none"
    oidc_provider: Optional[str] = None            # name of an [auth.<provider>] block in secrets.toml, or None for [auth]
    allow_anonymous_projects: bool = True
    max_projects_per_user: int = Field(default=50, ge=1)
    max_items_per_project: int = Field(default=1000, ge=10)
    max_note_chars: int = Field(default=10000, ge=100)
    anonymous_retention_days: int = Field(default=365, ge=7)
    create_rate_per_hour: int = Field(default=10, ge=1)   # per browser session


class Evaluation(_Section):
    enabled: bool = True
    eval_mode_env: str = "SCHOLARGRID_EVAL_MODE"
    admin_emails_env: str = "SCHOLARGRID_ADMIN_EMAILS"
    overlap_share: float = Field(default=0.2, ge=0, le=1)     # items rated by >= 2 raters
    sample_seed: int = 7
    report_dir: str = "reports/eval"


class Telemetry(_Section):
    enabled: bool = True
    store_queries: bool = False          # raw query text only with this explicitly on
    retention_days: int = Field(default=180, ge=7)
    sample_rate: float = Field(default=1.0, ge=0, le=1)


class ConfigModel(_Section):
    pipeline_version: str = "2.0.0"
    seed: int = 42
    environment: Literal["development", "production"] = "development"
    paths: Paths = Field(default_factory=Paths)
    data: Data = Field(default_factory=Data)
    storage: Storage = Field(default_factory=Storage)
    quality: Quality = Field(default_factory=Quality)
    enrich: Enrich = Field(default_factory=Enrich)
    graph: Graph = Field(default_factory=Graph)
    embedding: Embedding = Field(default_factory=Embedding)
    index: Index = Field(default_factory=Index)
    reduce: Reduce = Field(default_factory=Reduce)
    cluster: Cluster = Field(default_factory=Cluster)
    labeling: Labeling = Field(default_factory=Labeling)
    landscape: Landscape = Field(default_factory=Landscape)
    growth: Growth = Field(default_factory=Growth)
    search: Search = Field(default_factory=Search)
    sparse: Sparse = Field(default_factory=Sparse)
    validation: Validation = Field(default_factory=Validation)
    release: Release = Field(default_factory=Release)
    monitoring: Monitoring = Field(default_factory=Monitoring)
    app: App = Field(default_factory=App)
    llm: Llm = Field(default_factory=Llm)
    scope: Scope = Field(default_factory=Scope)
    fulltext: Fulltext = Field(default_factory=Fulltext)
    extract: Extract = Field(default_factory=Extract)
    foundations: Foundations = Field(default_factory=Foundations)
    workspace: Workspace = Field(default_factory=Workspace)
    gaps: Gaps = Field(default_factory=Gaps)
    plan: Plan = Field(default_factory=Plan)
    persistence: Persistence = Field(default_factory=Persistence)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    telemetry: Telemetry = Field(default_factory=Telemetry)


class ConfigError(ValueError):
    """Raised when a configuration file does not match the schema."""


def validate_config(raw: dict) -> dict:
    """Validate ``raw`` and return it with defaults filled in."""
    try:
        model = ConfigModel.model_validate(raw)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"  - {loc}: {err['msg']}")
        raise ConfigError("Invalid configuration:\n" + "\n".join(lines)) from None
    return model.model_dump()
