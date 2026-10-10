"""Pydantic schema for ``configs/*.yaml``.

Validation runs on every ``load_config`` call so a misspelt key or a wrong type
fails immediately with a readable message instead of deep inside a stage.
Unknown keys are rejected inside known sections.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


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
    live_sources: List[Literal["arxiv", "openalex", "semantic_scholar"]] = Field(
        default_factory=lambda: ["arxiv", "openalex", "semantic_scholar"])
    live_per_source: int = Field(default=200, ge=1)
    live_deadline_s: float = Field(default=10.0, gt=0)
    s2_budget_s: float = Field(default=6.0, gt=0)
    live_cache_hours: float = Field(default=24.0, ge=0)
    live_embed_max: int = Field(default=300, ge=0)
    live_embed_budget_s: float = Field(default=4.0, ge=0)
    cooldown_s: float = Field(default=60.0, ge=0)


class Sparse(_Section):
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


class Monitoring(_Section):
    json_logs: bool = False
    stale_after_days: int = 14


class App(_Section):
    max_graph_nodes: int = Field(default=20000, ge=100)
    max_area_papers: int = Field(default=400, ge=10)
    pre_warm: bool = True
    allow_grow: bool = True


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
