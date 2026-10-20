"""Topic workspace data model (JSON-serializable; cached on disk and in session state)."""
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

    def subtopic(self, sid: int) -> Optional[Subtopic]:
        return next((s for s in self.subtopics if s.id == sid), None)

    def papers_in(self, sid: int) -> List[WsPaper]:
        return [p for p in self.papers if p.subtopic == sid or p.micro == sid]

    def bundle_indices(self, sid: Optional[int] = None) -> List[int]:
        papers = self.papers if sid is None else self.papers_in(sid)
        return [int(p.bundle_index) for p in papers if p.bundle_index is not None]
