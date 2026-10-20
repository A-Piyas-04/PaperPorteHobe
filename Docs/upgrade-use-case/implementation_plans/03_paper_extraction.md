# Plan 03: Per-paper structured extraction (new stage `extract`)

> **Prerequisites:** plans 01 and 02 are merged. `data/fulltext/` holds sections and cue sentences for in-scope papers.
> **Produces:** for each in-scope paper, a validated `PaperExtraction` record: tasks, methods (with role), datasets, metrics, domains, languages, modalities, contribution types, evaluation profile, findings, and **stated open problems with verbatim, verified quotes**. Runs on the **free Gemini tier**: several papers are packed into one request, and the stage stops cleanly when the daily quota is used up and **resumes on the next run**. A heuristic fallback needs no LLM. Facet values are canonicalized across papers. Exploded tables ship in the bundle.
> **Why:** every gap detector in plan 06 works on these structured facts. Quote verification is the anti-hallucination foundation: a quote that is not literally in the paper never reaches a user.

---

## 1. Prompt to paste into Cursor

```text
Implement plan file 03_paper_extraction.md. Follow 00_INDEX_READ_FIRST.md and the LLM contract from
plan 01 (only scholargrid.llm talks to LLM providers; structured outputs via pydantic models; llm.structured_many
for bulk, packed, quota-aware).
Order: §3 config → §4 schemas → §5 document builder → §6 prompt → §7 heuristic extractor + fake handler
→ §8 quote verification → §9 store → §10 run_extract (packed LLM calls, quota stop/resume, heuristic fill) →
§11 canonicalization → §12 explode + bundle files → §13 pipeline + bundle loading → §14 CLI → §15 tests.
Run ruff, mypy and pytest after each section. Finish with §16.
```

---

## 2. Data flow

```
enriched df ──select_scope──► papers in scope
                                  │
data/fulltext (sections, cues) ───┤  build_document(): title, abstract, budgeted sections with [S#] markers,
                                  │                    detected cue sentences
                                  ▼
          ┌───────────── mode ─────────────┐
          │ llm (packed, quota-aware) │ heuristic │   (auto: llm if available, else heuristic; quota used up → heuristic now, upgrade later)
          └──────────────┬─────────────────┘
                         ▼
             PaperExtraction (pydantic, clipped)
                         │  verify_quotes(): exact → fuzzy (rapidfuzz alignment) → replace with exact span; drop unverifiable
                         ▼
           ExtractionStore  data/extract/extractions.parquet  (incremental by input hash + prompt version)
                         │
       analyze stage ──► explode + canonicalize (embeddings + string guard + acronyms)
                         ▼
 bundle: paper_facets.parquet, problems.parquet, findings.parquet, paper_profile.parquet, facet_vocab.json
```

---

## 3. Config

### 3.1 `scholargrid/schema.py`

```python
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
```

Add `extract: Extract = Field(default_factory=Extract)` to `ConfigModel`.

### 3.2 `configs/config.yaml`

```yaml
# ---------------------------------------------------------------------------
# Per-paper extraction (plan 03): facets + authors' own open problems (with
# verified verbatim quotes). auto = LLM when llm.provider is live/fake, else heuristics.
# ---------------------------------------------------------------------------
extract:
  enabled: true
  mode: auto
  store_dir: data/extract
  max_input_chars: 24000
  section_budget: {abstract: 3000, introduction: 3500, experiments: 3500, results: 2500,
                   discussion: 3000, limitations: 4000, future_work: 3000,
                   threats_to_validity: 2000, conclusion: 3000, broader_impact: 1500}
  max_cue_sentences: 25
  abstract_only_fallback: true
  max_llm_attempts: 3
  quote_min_ratio: 90
  max_problems_per_paper: 6
  max_findings_per_paper: 5
  on_quota: heuristic               # quota used up mid-run: heuristic (fill in now, upgrade on a later run) | skip | stop
  canonical_min_cosine: 0.88        # facet strings merge only if BOTH embedding and
  canonical_min_string_ratio: 80    # string similarity agree (BERT != RoBERTa)
```

`tests/conftest.py::make_config`: `raw["extract"]["store_dir"] = str(tmp_path / "extract")`.
`scholargrid/config.py`: add a property `extract_dir` (like `fulltext_dir`).

---

## 4. Schemas: `scholargrid/extract/schemas.py`

LLM-output models follow the plan 01 rule: **no length/number constraints in `Field`**. Validators clip instead.

```python
"""PaperExtraction: what the LLM (or the heuristic) returns for one paper."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Modality = Literal["text", "image", "video", "audio", "speech", "code", "tabular", "graph",
                   "time_series", "3d", "multimodal", "other"]
ContributionType = Literal["new_method", "benchmark_or_dataset", "empirical_study", "survey", "theory",
                           "system_or_tool", "position", "reproduction", "application"]
ProblemType = Literal["limitation_of_this_work", "future_work", "open_challenge_in_field",
                      "negative_result", "unexplored_setting"]
Direction = Literal["improves", "worsens", "no_effect", "mixed"]
Compute = Literal["unknown", "laptop", "single_gpu", "multi_gpu", "large_cluster"]
Role = Literal["proposed", "baseline", "used"]


def _clip(v: object, n: int) -> str:
    s = " ".join(str(v or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _clean_list(values: object, n_items: int, n_chars: int = 80) -> List[str]:
    out, seen = [], set()
    for v in values or []:          # type: ignore[union-attr]
        s = _clip(v, n_chars).strip(" .;,")
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:n_items]


class MethodMention(BaseModel):
    name: str = Field(description="Canonical method/model name as commonly written, e.g. 'LoRA', "
                                  "'speculative decoding', 'BERT'. Not a description.")
    role: Role = Field(description="proposed = introduced or extended by this paper; baseline = compared "
                                   "against; used = applied as a tool/component")

    @field_validator("name", mode="before")
    @classmethod
    def _n(cls, v):
        return _clip(v, 80)


class StatedProblem(BaseModel):
    statement: str = Field(description="Self-contained paraphrase (<= 40 words) of an open problem the "
                                       "AUTHORS state. Name the method/task explicitly; never write "
                                       "'our method'.")
    quote: str = Field(description="Exact sentence(s) copied character-for-character from the paper that "
                                   "state this problem (<= 300 characters).")
    section: str = Field(description="The [S#] marker of the section the quote comes from, e.g. 'S5'.")
    problem_type: ProblemType
    specificity: int = Field(description="1 = vague ('more work is needed'), 2 = names the issue, "
                                         "3 = concrete and testable (setting, data or metric named)")

    @field_validator("statement", mode="before")
    @classmethod
    def _s(cls, v):
        return _clip(v, 320)

    @field_validator("quote", mode="before")
    @classmethod
    def _q(cls, v):
        return _clip(v, 400)

    @field_validator("specificity", mode="before")
    @classmethod
    def _sp(cls, v):
        try:
            return min(3, max(1, int(v)))
        except (TypeError, ValueError):
            return 1


class Finding(BaseModel):
    claim: str = Field(description="One empirical result in <= 30 words")
    task: str
    dataset: str = Field(description="Dataset/benchmark name or '' if not stated")
    metric: str = Field(description="Metric name or '' if not stated")
    direction: Direction = Field(description="Effect of the paper's method/intervention vs the comparison")
    compared_to: str = Field(description="What it is compared against, or ''")

    @field_validator("claim", "task", "dataset", "metric", "compared_to", mode="before")
    @classmethod
    def _c(cls, v):
        return _clip(v, 200)


class EvalProfile(BaseModel):
    real_world: Optional[bool] = Field(description="Evaluated on real deployment / in-the-wild data")
    human_eval: Optional[bool] = Field(description="Includes human evaluation or a user study")
    robustness: Optional[bool] = Field(description="Tests robustness / OOD / adversarial conditions")
    efficiency_reported: Optional[bool] = Field(description="Reports latency, memory or compute cost")
    low_resource: Optional[bool] = Field(description="Evaluates low-resource data, languages or settings")
    multilingual: Optional[bool] = Field(description="Evaluates more than one natural language")


class PaperExtraction(BaseModel):
    one_line_summary: str = Field(description="What the paper does, <= 30 words, plain English")
    tasks: List[str] = Field(description="Research tasks/problems addressed, short noun phrases, e.g. "
                                         "'machine translation', 'code generation'")
    methods: List[MethodMention]
    datasets: List[str] = Field(description="Named datasets/benchmarks actually used")
    metrics: List[str]
    domains: List[str] = Field(description="Application domains, e.g. 'healthcare', 'legal'; ['general'] if none")
    languages: List[str] = Field(description="Natural languages of the data, e.g. 'English', 'Bengali'; [] if n/a")
    modalities: List[Modality]
    contribution_types: List[ContributionType]
    evaluation: EvalProfile
    open_problems: List[StatedProblem] = Field(description="Problems the authors say remain open, "
                                                           "their own limitations, or future work. [] if none.")
    findings: List[Finding]
    code_url: Optional[str] = Field(description="Code repository URL if stated, else null")
    data_released: Optional[bool]
    compute: Compute

    @field_validator("one_line_summary", mode="before")
    @classmethod
    def _sum(cls, v):
        return _clip(v, 240)

    @field_validator("tasks", "datasets", "metrics", "domains", "languages", mode="before")
    @classmethod
    def _lists(cls, v):
        return _clean_list(v, 12)

    @field_validator("methods", mode="before")
    @classmethod
    def _methods(cls, v):
        return list(v or [])[:15]

    @field_validator("open_problems", mode="before")
    @classmethod
    def _probs(cls, v):
        return list(v or [])[:8]

    @field_validator("findings", mode="before")
    @classmethod
    def _finds(cls, v):
        return list(v or [])[:6]

    @field_validator("code_url", mode="before")
    @classmethod
    def _url(cls, v):
        s = str(v or "").strip()
        return s if s.startswith(("http://", "https://")) else None
```

---

## 5. Document builder: `scholargrid/extract/document.py`

```python
"""Builds the text the LLM reads for one paper: budgeted sections with [S#] markers + cue sentences."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

_ORDER = ["abstract", "introduction", "experiments", "results", "discussion", "limitations",
          "future_work", "threats_to_validity", "conclusion", "broader_impact"]


@dataclass
class PaperDocument:
    arxiv_id: str
    text: str                     # what the model sees
    sources: Dict[str, str]       # "S5" -> full stored section text (for quote verification)
    anchors: Dict[str, str]       # "S5" -> HTML anchor (e.g. "S5.SS1")
    kinds: Dict[str, str]         # "S5" -> section kind
    has_fulltext: bool

    @property
    def input_hash(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:16]


def build_document(arxiv_id: str, title: str, abstract: str, sections: pd.DataFrame, cues: pd.DataFrame,
                   budget: Dict[str, int], max_chars: int, max_cues: int) -> PaperDocument:
    parts: List[str] = [f"TITLE: {title}"]
    sources, anchors, kinds = {}, {}, {}
    secs = sections.sort_values("idx") if len(sections) else sections
    have_abstract = len(secs) and (secs["kind"] == "abstract").any()
    if not have_abstract:
        parts.append(f"[S0] ABSTRACT\n{abstract}")
        sources["S0"], anchors["S0"], kinds["S0"] = abstract, "abstract", "abstract"
    for kind in _ORDER:
        for r in secs[secs["kind"] == kind].itertuples() if len(secs) else []:
            if not r.text:
                continue
            label = f"S{int(r.idx)}"
            body = str(r.text)[: int(budget.get(kind, 2000))]
            parts.append(f"[{label}] {kind.upper()}: {r.heading}\n{body}")
            sources[label], anchors[label], kinds[label] = str(r.text), str(r.anchor or ""), kind
    if max_cues and len(cues):
        lines = [f"- [S{int(c.section_idx)}] {c.sentence}" for c in cues.head(max_cues).itertuples()]
        parts.append("GAP-CUE SENTENCES DETECTED AUTOMATICALLY (may be incomplete):\n" + "\n".join(lines))
    text = "\n\n".join(parts)[:max_chars]
    return PaperDocument(arxiv_id, text, sources, anchors, kinds, has_fulltext=bool(len(secs)))
```

---

## 6. Prompt: `scholargrid/extract/prompts.py`

Register it, and add `"scholargrid.extract.prompts"` to `FEATURE_PROMPT_MODULES` in `scholargrid/llm/__init__.py`.

```python
"""Prompts for per-paper extraction."""
from __future__ import annotations

from ..llm.fakes import register_fake
from ..llm.prompts import Prompt, register

EXTRACT_SYSTEM = """You extract structured facts from computer-science research papers for a tool that helps \
early-career researchers find open research problems.

Read the paper excerpt and fill the schema. Rules:
1. Only record what the paper states. If something is not stated, use an empty list, null, \
"unknown" or "" as the schema allows. Never infer from your own knowledge of the field.
2. tasks: the research problems addressed, as short lowercase noun phrases ("machine translation", \
"text-to-sql", "image segmentation"). Do not list generic words like "learning" or "performance".
3. methods: named techniques/models. role=proposed only for what this paper introduces or extends; \
baseline for what it compares against; used for components it relies on.
4. datasets / metrics: the names as written ("SQuAD 2.0", "BLEU"). Only those actually used.
5. open_problems: the most important part. Include limitations of this work, future work the authors \
propose, and open challenges they say remain in the field. For each:
   - statement: a self-contained paraphrase a newcomer can understand without the paper. Name the \
method, task, data or setting. Never write "our method" or "this paper".
   - quote: copy the supporting sentence(s) EXACTLY, character for character, from the excerpt, \
including its capitalisation and punctuation. At most 300 characters. If you cannot quote it exactly, \
leave the problem out.
   - section: the [S#] marker of the section the quote comes from.
   - specificity: 1 vague, 2 names the issue, 3 concrete and testable.
   Do not include generic statements such as "more research is needed" unless they name a specific issue.
6. findings: up to 5 main empirical results, each tied to a task, dataset and metric when stated.
7. evaluation: true/false only when the paper makes it clear; otherwise null.
8. The "GAP-CUE SENTENCES" list is a hint from a keyword detector. It can contain false positives \
and miss things."""

register(Prompt(
    name="extract_paper",
    version="1",
    system=EXTRACT_SYSTEM,
    user="arXiv id: $arxiv_id\n\n$document\n\nExtract the structured record for this paper.",
    packable=True,      # llm.tasks.extract.pack_size papers per request (default 4)
))


@register_fake("extract_paper")
def _fake_extract(v: dict) -> dict:
    from .heuristic import heuristic_extract_from_text

    return heuristic_extract_from_text(str(v["document"])).model_dump()
```

The caller wraps the document: `document=wrap_untrusted("paper", doc.text)`. Do not wrap inside the template, so the fake handler and the heuristic see the same text. The fake handler strips the tags (see §7).

---

## 7. Heuristic extractor: `scholargrid/extract/heuristic.py`

Used when no LLM is available, when the daily free quota is used up (filled in, then upgraded on a later run), by `FakeLLM` in tests, and for abstract-only papers when `mode: heuristic`. It is deliberately simple, transparent and conservative.

```python
"""No-LLM extraction from the same [S#]-marked document the LLM would see."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..fulltext.sections import CUE_PATTERNS, split_sentences
from .schemas import EvalProfile, MethodMention, PaperExtraction, StatedProblem

_SECTION = re.compile(r"^\[(S\d+)\]\s+([A-Z_]+)(?::\s*(.*))?$", re.M)
_METRICS = ["accuracy", "f1", "bleu", "rouge", "exact match", "perplexity", "auc", "map", "miou", "psnr",
            "ssim", "fid", "wer", "cer", "latency", "throughput", "recall@", "precision", "ndcg", "mrr",
            "pass@", "success rate", "spearman", "pearson"]
_LANGS = ["English", "Chinese", "Bengali", "Bangla", "Hindi", "Arabic", "Spanish", "French", "German",
          "Japanese", "Korean", "Swahili", "Urdu", "Portuguese", "Russian", "Indonesian", "Vietnamese"]
_MODALITY = {"image": r"\bimages?\b|\bvision\b|\bvisual\b", "video": r"\bvideos?\b", "speech": r"\bspeech\b|\basr\b",
             "audio": r"\baudio\b", "code": r"\bcode\b|\bprogram", "graph": r"\bgraphs?\b|\bgnn",
             "tabular": r"\btabular\b", "time_series": r"time[- ]series", "3d": r"\b3d\b|point cloud",
             "text": r"\btext\b|\blanguage\b|\bnlp\b"}
_DATASET = re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:[ \-]?\d+(?:\.\d+)?)?)\s+(?:dataset|benchmark|corpus)\b"
                      r"|\b(Synth[A-Za-z]+)\b")
_ON_DATASETS = re.compile(r"\bon\s+((?:[A-Z][\w\-.]+(?:,\s*|\s+and\s+)?){1,4})")
_ACRONYM = re.compile(r"\b([A-Z][A-Za-z]*[A-Z][A-Za-z0-9\-]*)\b")
_STOP_ACR = {"AI", "ML", "NLP", "CV", "GPU", "CPU", "API", "URL", "SOTA", "TITLE", "ABSTRACT", "GAP", "CUE",
             "IID", "OOD", "US", "UK", "EU"}
_TYPE_BY_CUE = {"future_work": "future_work", "left_for_future": "future_work", "we_plan": "future_work",
                "direction": "future_work", "open_question": "open_challenge_in_field",
                "open_problem": "open_challenge_in_field", "not_explored": "unexplored_setting",
                "little_work": "unexplored_setting"}


def _sections(text: str) -> List[Tuple[str, str, str, str]]:
    """[(label, kind, heading, body)] from an [S#]-marked document (tags stripped)."""
    text = re.sub(r"^<paper>\n|\n</paper>$", "", text.strip())
    marks = list(_SECTION.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end]
        body = body.split("GAP-CUE SENTENCES DETECTED")[0].strip()
        out.append((m.group(1), m.group(2).lower(), m.group(3) or "", body))
    return out


def heuristic_extract_from_text(document: str) -> PaperExtraction:
    secs = _sections(document)
    title = (re.search(r"^TITLE:\s*(.+)$", document, re.M) or [None, ""])[1] if "TITLE:" in document else ""
    full = " ".join(b for *_, b in secs)
    low = full.lower()

    problems: List[StatedProblem] = []
    for label, kind, _, body in secs:
        if kind not in {"limitations", "future_work", "discussion", "conclusion", "threats_to_validity"}:
            continue
        for _, sent in split_sentences(body):
            cue = next((n for n, rx in CUE_PATTERNS if rx.search(sent)), None)
            if cue is None and kind not in {"limitations", "future_work"}:
                continue
            ptype = _TYPE_BY_CUE.get(cue or "", "limitation_of_this_work" if kind == "limitations" else "future_work")
            subject = title.split(":")[0].strip() or "the approach"
            statement = re.sub(r"\b(our|this)\s+(approach|method|model|work|framework)\b", subject, sent, flags=re.I)
            problems.append(StatedProblem(statement=statement, quote=sent[:300], section=label,
                                          problem_type=ptype, specificity=2))
    datasets = sorted({(m.group(1) or m.group(2)) for m in _DATASET.finditer(full) if (m.group(1) or m.group(2))})
    for m in _ON_DATASETS.finditer(full):
        datasets += [d.strip(" ,.") for d in re.split(r",|\band\b", m.group(1)) if d.strip(" ,.").startswith("Synth")]
    methods = [MethodMention(name=title.split(":")[0].strip(), role="proposed")] if ":" in title else []
    methods += [MethodMention(name=a, role="used") for a in sorted(set(_ACRONYM.findall(full)) - _STOP_ACR)[:8]]
    task = (re.search(r"\bfor\s+([a-z][a-z \-]{3,40})$", title.lower()) or [None, ""])[1] if title else ""
    return PaperExtraction(
        one_line_summary=(split_sentences(full)[0][1] if full else title)[:240],
        tasks=[task] if task else [],
        methods=methods,
        datasets=sorted(set(datasets)),
        metrics=[m for m in _METRICS if m in low],
        domains=["general"],
        languages=[lang for lang in _LANGS if re.search(rf"\b{lang}\b", full)],
        modalities=[k for k, rx in _MODALITY.items() if re.search(rx, low)][:3] or ["other"],
        contribution_types=["survey"] if re.search(r"\bsurvey|review\b", title.lower()) else ["new_method"],
        evaluation=EvalProfile(
            real_world=True if re.search(r"real[- ]world|in the wild|deployed", low) else None,
            human_eval=True if re.search(r"human evaluation|user study|annotators", low) else None,
            robustness=True if re.search(r"robust|adversarial|distribution shift|out-of-distribution", low) else None,
            efficiency_reported=True if re.search(r"latency|throughput|memory footprint|flops", low) else None,
            low_resource=True if "low-resource" in low or "low resource" in low else None,
            multilingual=True if re.search(r"multilingual|cross-lingual", low) else None),
        open_problems=problems,
        findings=[],
        code_url=(re.search(r"https?://github\.com/[\w\-./]+", full) or [None])[0],
        data_released=None,
        compute="multi_gpu" if re.search(r"\b\d+\s*(x\s*)?(a100|h100|v100|gpus)\b", low) else
                ("single_gpu" if re.search(r"\b(a100|h100|v100|rtx|gpu)\b", low) else "unknown"),
    )
```

---

## 8. Quote verification: `scholargrid/extract/verify.py`

The quote shown to a user must be **literally** in the paper. The model's quote is located in the cited section (falling back to the whole document). It is then **replaced by the exact source span**, so tiny transcription differences (whitespace, curly quotes) are repaired. Problems that cannot be located are **dropped**.

```python
"""Verify that every stated-problem quote exists in the source text."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .document import PaperDocument
from .schemas import PaperExtraction, StatedProblem

_TRANS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " "})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_TRANS)).strip()


@dataclass
class VerifiedProblem:
    problem: StatedProblem
    quote: str               # exact span from the source
    section_label: str
    ratio: float             # 100 = exact


def locate(quote: str, source: str, min_ratio: int) -> Optional[tuple[str, float]]:
    q, s = _norm(quote).rstrip("…"), _norm(source)
    if not q or not s:
        return None
    i = s.find(q)
    if i >= 0:
        return s[i:i + len(q)], 100.0
    from rapidfuzz import fuzz

    al = fuzz.partial_ratio_alignment(q, s)
    if al is None or al.score < min_ratio:
        return None
    return s[al.dest_start:al.dest_end].strip(), float(al.score)


def verify_quotes(ext: PaperExtraction, doc: PaperDocument, min_ratio: int, max_problems: int
                  ) -> tuple[List[VerifiedProblem], int]:
    """Returns (verified problems, number dropped)."""
    kept: List[VerifiedProblem] = []
    dropped = 0
    whole = " ".join(doc.sources.values()) or doc.text
    for p in ext.open_problems:
        label = p.section if p.section in doc.sources else ""
        hit = locate(p.quote, doc.sources[label], min_ratio) if label else None
        if hit is None:
            hit = locate(p.quote, whole, min_ratio)
            label = label or next((k for k, v in doc.sources.items() if hit and _norm(hit[0]) in _norm(v)), "")
        if hit is None or len(hit[0]) < 20:
            dropped += 1
            continue
        kept.append(VerifiedProblem(p, hit[0][:300], label, hit[1]))
    kept.sort(key=lambda v: (-v.problem.specificity, -v.ratio))
    return kept[:max_problems], dropped
```

---

## 9. Store: `scholargrid/extract/store.py`

```python
"""data/extract/extractions.parquet - one latest row per paper.

Columns: arxiv_id, input_hash, prompt_version, mode (llm | llm_abstract | heuristic | heuristic_abstract),
model, extracted_at, payload (PaperExtraction JSON), problems (verified problems JSON),
n_problems_raw, n_problems_verified, cost_usd (always 0 on free tiers), llm_attempts
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, List

import pandas as pd

from ..config import Config
from ..utils import ensure_dir

COLUMNS = ["arxiv_id", "input_hash", "prompt_version", "mode", "model", "extracted_at", "payload",
           "problems", "n_problems_raw", "n_problems_verified", "cost_usd", "llm_attempts"]


class ExtractionStore:
    def __init__(self, cfg: Config):
        self.dir = ensure_dir(cfg.extract_dir)
        self.path = os.path.join(self.dir, "extractions.parquet")
        self.df = pd.read_parquet(self.path) if os.path.exists(self.path) else pd.DataFrame(columns=COLUMNS)
        self._pending: List[Dict] = []

    def is_current(self, arxiv_id: str, input_hash: str, prompt_version: str, want_llm: bool) -> bool:
        rows = self.df[self.df["arxiv_id"] == arxiv_id]
        if rows.empty:
            return False
        r = rows.iloc[-1]
        if r["input_hash"] != input_hash or r["prompt_version"] != prompt_version:
            return False
        return not (want_llm and str(r["mode"]).startswith("heuristic"))   # upgrade heuristic -> llm

    def add(self, **row) -> None:
        row["extracted_at"] = time.time()
        for key in ("payload", "problems"):
            if not isinstance(row[key], str):
                row[key] = json.dumps(row[key])
        self._pending.append(row)

    def flush(self) -> None:
        if not self._pending:
            return
        new = pd.DataFrame(self._pending, columns=COLUMNS)
        self.df = (pd.concat([self.df, new], ignore_index=True)
                   .sort_values("extracted_at").drop_duplicates("arxiv_id", keep="last"))
        tmp = self.path + ".tmp"
        self.df.to_parquet(tmp, index=False)
        os.replace(tmp, self.path)
        self._pending = []

    def latest(self, ids: Iterable[str]) -> pd.DataFrame:
        return self.df[self.df["arxiv_id"].isin(set(ids))].reset_index(drop=True)
```

---

## 10. `run_extract`: `scholargrid/extract/run.py`

Decision logic (free-tier aware):
- `mode=auto` → LLM if `get_llm(cfg).available()`, else heuristic.
- Items to do = in-scope papers whose stored row is not current: a changed input hash or prompt version, or a **heuristic row while an LLM is available** (this is how papers filled in heuristically yesterday get upgraded today). Skip papers whose `llm_attempts ≥ max_llm_attempts`.
- Papers are processed in `select_scope` priority order (most cited first), so limited quota goes to the most useful papers.
- LLM calls go through `llm.structured_many("extract", "extract_paper", PaperExtraction, items)`. It packs `llm.tasks.extract.pack_size` papers per request (default 4), retries missing items alone, and on `QuotaExhausted` returns that error for every remaining item.
- Remaining items after a quota stop, per `extract.on_quota`:
  - `heuristic` (default): record a heuristic extraction now. It is upgraded on a later run.
  - `skip`: leave them for the next run.
  - `stop`: re-raise.
- Per-item `LLMRefusal` / `LLMInvalidOutput` / `LLMTruncated` → heuristic, and `llm_attempts += 1`.
- `BudgetExceeded` (`llm.budget.offline_max_requests`) is treated like a quota stop.
- Stats report `quota_stopped`, `pending_llm` (papers still waiting for an LLM pass) and `requests_used`, so the runbook can show progress across days.

Add a column `llm_attempts` (int) to `ExtractionStore.COLUMNS` (§9), carry it in `add()`, and add `ExtractionStore.attempts(arxiv_id) -> int` (0 when there is no row).

```python
"""run_extract(df, cfg) -> stats. Incremental, quota-aware, resumable across days; never drops a paper."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..config import Config
from ..fulltext import FulltextStore
from ..llm import BudgetExceeded, LLMError, QuotaExhausted, get_llm
from ..llm.prompts import PROMPTS, wrap_untrusted
from ..scope import select_scope
from ..utils import get_logger
from .document import PaperDocument, build_document
from .heuristic import heuristic_extract_from_text
from .schemas import PaperExtraction
from .store import ExtractionStore
from .verify import verify_quotes

log = get_logger("extract")
PROMPT = "extract_paper"
CHUNK = 40          # papers per flush (a crash or Ctrl-C keeps finished work)


def _documents(scope: pd.DataFrame, cfg: Config) -> List[PaperDocument]:
    e = cfg["extract"]
    store = FulltextStore(cfg)
    ids = scope["arxiv_id"].astype(str).tolist()
    secs, cues = store.sections(ids), store.cues(ids)
    by_sec = {k: g for k, g in secs.groupby("arxiv_id")} if len(secs) else {}
    by_cue = {k: g for k, g in cues.groupby("arxiv_id")} if len(cues) else {}
    docs = []
    for r in scope.itertuples():                      # keeps select_scope priority order
        aid = str(r.arxiv_id)
        s = by_sec.get(aid, pd.DataFrame(columns=["idx", "kind", "heading", "text", "anchor"]))
        if s.empty and not e["abstract_only_fallback"]:
            continue
        docs.append(build_document(aid, str(r.title), str(r.abstract), s, by_cue.get(aid, pd.DataFrame()),
                                   e["section_budget"], int(e["max_input_chars"]), int(e["max_cue_sentences"])))
    return docs


def _record(store, doc, ext, mode, model, cfg, attempts) -> tuple[int, int]:
    e = cfg["extract"]
    verified, dropped = verify_quotes(ext, doc, int(e["quote_min_ratio"]), int(e["max_problems_per_paper"]))
    problems = [{"statement": v.problem.statement, "quote": v.quote, "section": v.section_label,
                 "anchor": doc.anchors.get(v.section_label, ""), "section_kind": doc.kinds.get(v.section_label, ""),
                 "problem_type": v.problem.problem_type, "specificity": v.problem.specificity,
                 "verified_ratio": v.ratio} for v in verified]
    suffix = "" if doc.has_fulltext else "_abstract"
    store.add(arxiv_id=doc.arxiv_id, input_hash=doc.input_hash, prompt_version=PROMPTS[PROMPT].version,
              mode=mode + suffix, model=model, payload=ext.model_dump(mode="json"), problems=problems,
              n_problems_raw=len(ext.open_problems), n_problems_verified=len(problems), cost_usd=0.0,
              llm_attempts=attempts)
    return len(problems), dropped


def run_extract(df: pd.DataFrame, cfg: Config) -> Dict:
    e = cfg["extract"]
    stats = {"enabled": bool(e["enabled"]), "in_scope": 0, "done_now": 0, "llm": 0, "heuristic": 0,
             "problems_verified": 0, "problems_dropped": 0, "quota_stopped": False, "pending_llm": 0,
             "requests_used": 0}
    if not e["enabled"]:
        return stats
    scope = select_scope(df, cfg)
    stats["in_scope"] = int(len(scope))
    llm = get_llm(cfg)
    use_llm = e["mode"] == "llm" or (e["mode"] == "auto" and llm.available())
    store = ExtractionStore(cfg)
    version = PROMPTS[PROMPT].version
    docs = [d for d in _documents(scope, cfg)
            if not store.is_current(d.arxiv_id, d.input_hash, version, use_llm)
            and store.attempts(d.arxiv_id) < int(e["max_llm_attempts"])]
    log.info("Extraction: %d in scope, %d to process (%s).", len(scope), len(docs), "llm" if use_llm else "heuristic")
    ledger = getattr(llm, "ledger", None)
    req_before = getattr(ledger, "requests", 0)

    def heuristic(doc: PaperDocument, attempts: int) -> None:
        v, d = _record(store, doc, heuristic_extract_from_text(doc.text), "heuristic", "heuristic", cfg, attempts)
        stats["heuristic"] += 1
        stats["problems_verified"] += v
        stats["problems_dropped"] += d

    def after_quota(doc: PaperDocument) -> None:
        stats["quota_stopped"] = True
        stats["pending_llm"] += 1
        if e["on_quota"] == "heuristic":
            heuristic(doc, store.attempts(doc.arxiv_id))      # upgraded by a later run

    if not use_llm:
        for doc in docs:
            heuristic(doc, store.attempts(doc.arxiv_id))
    else:
        by_id = {d.arxiv_id: d for d in docs}
        items = [(d.arxiv_id, {"arxiv_id": d.arxiv_id, "document": wrap_untrusted("paper", d.text)}) for d in docs]
        for start in range(0, len(items), CHUNK):
            chunk = items[start:start + CHUNK]
            if stats["quota_stopped"]:                         # quota gone: no more LLM calls this run
                for iid, _ in chunk:
                    after_quota(by_id[iid])
                continue
            try:
                results = llm.structured_many("extract", PROMPT, PaperExtraction, chunk)
            except BudgetExceeded as exc:
                results = {iid: QuotaExhausted(str(exc)) for iid, _ in chunk}
            for iid, _ in chunk:
                doc, res = by_id[iid], results.get(iid)
                if isinstance(res, QuotaExhausted):
                    if e["on_quota"] == "stop":
                        store.flush()
                        raise res
                    after_quota(doc)
                elif isinstance(res, LLMError) or res is None:
                    heuristic(doc, store.attempts(iid) + 1)
                else:
                    v, d = _record(store, doc, res.value, "llm", res.route, cfg, 0)
                    stats["llm"] += 1
                    stats["problems_verified"] += v
                    stats["problems_dropped"] += d
            store.flush()
    store.flush()
    stats["done_now"] = len(docs)
    stats["requests_used"] = getattr(ledger, "requests", 0) - req_before
    total = stats["problems_verified"] + stats["problems_dropped"]
    stats["quote_verified_rate"] = round(stats["problems_verified"] / total, 4) if total else None
    if stats["quota_stopped"]:
        log.warning("Free quota used up: %d papers wait for an LLM pass; rerun after the quota resets "
                    "(`python -m scholargrid.llm quota`).", stats["pending_llm"])
    return stats
```

## 11. Facet canonicalization: `scholargrid/extract/canonical.py`

Merging rule (precision over recall). Two surface forms of the same facet type merge when **all** of:
1. normalized forms are equal (lowercase, unicode dashes, articles, simple plural folding), **or**
2. one is an acronym whose mined long form equals the other (`acronyms.mine` over abstracts), **or**
3. embedding cosine ≥ `canonical_min_cosine` **and** `rapidfuzz.fuzz.token_set_ratio` ≥ `canonical_min_string_ratio`.

Rule 3 has both guards because embeddings put "BERT" next to "RoBERTa" and "BLEU" next to "ROUGE". Components come from union-find over candidate edges; candidates come from a top-k cosine search (`index.exact_topk`, k=10), so it never builds an n×n matrix.

```python
"""Merge facet surface forms ("LLMs", "large language models") into canonical values."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Mapping

import numpy as np

from ..index import exact_topk
from ..labeling import _stem

_ARTICLE = re.compile(r"^(the|a|an)\s+")


def normalize_value(v: str) -> str:
    s = unicodedata.normalize("NFKC", str(v)).lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"[\"'`()\[\]]", " ", s)
    s = _ARTICLE.sub("", " ".join(s.split())).strip(" .,;:")
    words = s.split()
    if words:
        words[-1] = _stem(words[-1])
    return " ".join(words)


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def canonicalize_values(raw_counts: Mapping[str, int], vectors: np.ndarray | None,
                        acronyms: Mapping[str, Mapping[str, int]], min_cos: float, min_ratio: int
                        ) -> Dict[str, str]:
    """{raw value -> canonical display value}. ``vectors`` rows align with ``list(raw_counts)``."""
    from rapidfuzz import fuzz

    raws = list(raw_counts)
    norms = [normalize_value(r) for r in raws]
    uf = _UF(len(raws))
    first: Dict[str, int] = {}
    for i, n in enumerate(norms):                       # rule 1
        if n in first:
            uf.union(i, first[n])
        else:
            first[n] = i
    long_of = {a: max(forms, key=forms.get) for a, forms in acronyms.items() if forms}
    for i, n in enumerate(norms):                       # rule 2
        lf = long_of.get(n)
        if lf:
            j = first.get(normalize_value(lf))
            if j is not None:
                uf.union(i, j)
    if vectors is not None and len(raws) > 1:           # rule 3
        idx, sims = exact_topk(vectors, vectors, min(10, len(raws) - 1), exclude_self=True)
        for i in range(len(raws)):
            for j, s in zip(idx[i], sims[i]):
                if s >= min_cos and fuzz.token_set_ratio(norms[i], norms[int(j)]) >= min_ratio:
                    uf.union(i, int(j))
    groups: Dict[int, List[int]] = {}
    for i in range(len(raws)):
        groups.setdefault(uf.find(i), []).append(i)
    out: Dict[str, str] = {}
    for members in groups.values():
        canon = max(members, key=lambda k: (raw_counts[raws[k]], -len(raws[k])))
        for k in members:
            out[raws[k]] = raws[canon]
    return out
```

`exact_topk(queries, matrix, k, exclude_self=True)` (existing, `scholargrid/index.py`) returns `(indices, sims)` sorted by similarity and never includes the row itself. Normalize vectors with `utils.l2_normalize` first.

---

## 12. Explode + bundle files: `scholargrid/extract/bundle.py`

```python
"""Turns stored extractions into the bundle tables the app and the gap engine read."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd

from ..acronyms import mine as mine_acronyms
from ..config import Config
from ..utils import get_logger, l2_normalize, save_json
from .canonical import canonicalize_values
from .store import ExtractionStore

log = get_logger("extract.bundle")
FACETS = ("task", "method", "dataset", "metric", "domain", "language", "modality", "contribution")
FILES = ("paper_facets.parquet", "problems.parquet", "findings.parquet", "paper_profile.parquet")


def explode(rows: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    facets, problems, findings, profiles = [], [], [], []
    for r in rows.itertuples():
        p = json.loads(r.payload)
        aid = str(r.arxiv_id)
        for facet, key in (("task", "tasks"), ("dataset", "datasets"), ("metric", "metrics"),
                           ("domain", "domains"), ("language", "languages"), ("modality", "modalities"),
                           ("contribution", "contribution_types")):
            facets += [{"arxiv_id": aid, "facet": facet, "value_raw": v, "role": ""} for v in p.get(key, [])]
        facets += [{"arxiv_id": aid, "facet": "method", "value_raw": m["name"], "role": m["role"]}
                   for m in p.get("methods", [])]
        for q in json.loads(r.problems):
            pid = hashlib.sha1(f"{aid}|{q['quote']}".encode()).hexdigest()[:12]
            problems.append({"problem_id": pid, "arxiv_id": aid, **q, "source_mode": r.mode})
        for f in p.get("findings", []):
            fid = hashlib.sha1(f"{aid}|{f['claim']}".encode()).hexdigest()[:12]
            findings.append({"finding_id": fid, "arxiv_id": aid, **f})
        ev = p.get("evaluation") or {}
        profiles.append({"arxiv_id": aid, "summary": p.get("one_line_summary", ""), "code_url": p.get("code_url") or "",
                         "data_released": p.get("data_released"), "compute": p.get("compute", "unknown"),
                         "mode": r.mode, "model": r.model, **{f"eval_{k}": v for k, v in ev.items()}})
    return {"paper_facets": pd.DataFrame(facets, columns=["arxiv_id", "facet", "value_raw", "role"]),
            "problems": pd.DataFrame(problems), "findings": pd.DataFrame(findings),
            "paper_profile": pd.DataFrame(profiles)}


def write_extraction_bundle(cfg: Config, df: pd.DataFrame, embedder, out_dir: str) -> Dict:
    rows = ExtractionStore(cfg).latest(df["arxiv_id"].astype(str))
    tables = explode(rows)
    facets = tables["paper_facets"]
    vocab: Dict[str, Dict] = {}
    acr = mine_acronyms(df["abstract"].astype(str))
    e = cfg["extract"]
    facets["value"] = facets["value_raw"]
    for facet in FACETS:
        sub = facets[facets["facet"] == facet]
        if sub.empty:
            continue
        counts = Counter(sub["value_raw"])
        vecs = None
        if embedder is not None and facet in ("task", "method", "dataset", "metric", "domain") and len(counts) > 1:
            vecs = l2_normalize(np.asarray(embedder.transform(list(counts)), dtype=np.float32))
        mapping = canonicalize_values(counts, vecs, acr, float(e["canonical_min_cosine"]),
                                      int(e["canonical_min_string_ratio"]))
        facets.loc[sub.index, "value"] = sub["value_raw"].map(mapping)
        agg: Dict[str, Dict] = {}
        for raw, canon in mapping.items():
            a = agg.setdefault(canon, {"count": 0, "aliases": []})
            a["count"] += counts[raw]
            if raw != canon:
                a["aliases"].append(raw)
        vocab[facet] = dict(sorted(agg.items(), key=lambda kv: -kv[1]["count"]))
    tables["paper_facets"] = facets.drop_duplicates(["arxiv_id", "facet", "value", "role"])
    for name, t in tables.items():
        t.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
    save_json(vocab, os.path.join(out_dir, "facet_vocab.json"))
    stats = {"papers_with_extraction": int(len(rows)), "problems": int(len(tables["problems"])),
             "facet_values": {f: len(v) for f, v in vocab.items()},
             "modes": rows["mode"].value_counts().to_dict() if len(rows) else {}}
    log.info("Extraction bundle: %s", stats)
    return stats
```

Problems also need their own **embeddings** for clustering (plan 06). Compute them here and save `problem_embeddings.npy` (float16, rows aligned with `problems.parquet`) using `embedder.transform(statements)`. If `embedder` is None, skip; plan 06 then embeds lazily.

---

## 13. Pipeline and bundle loading

### 13.1 `scholargrid/pipeline.py`

```python
STAGES = ["ingest", "enrich", "fulltext", "extract", "embed", "analyze", "validate", "publish"]
_PROGRESS["extract"] = (0.22, "Extracting methods, datasets and open problems…")


def stage_extract(ctx: Context) -> None:
    from .extract import run_extract

    with stage_timer("extract", ctx.timings) as info:
        stats = run_extract(ctx.frame("enriched"), ctx.cfg)
        ctx.save_json("extract_stats", stats)
        info.update(**{k: stats[k] for k in ("in_scope", "done_now", "llm", "heuristic", "pending_llm", "requests_used")})
```

In `stage_analyze`, right after `save_bundle(...)`:

```python
    with stage_timer("extraction_bundle", t) as info:
        from .extract.bundle import write_extraction_bundle

        info.update(**write_extraction_bundle(cfg, df, embedder, cfg.processed_dir))
```

And put `"extract": ctx.json("extract_stats", {})` into `meta`. `meta` is built before `save_bundle`, so add the extraction-bundle stats into `meta` and re-save `meta.json` afterwards with `save_json(meta, os.path.join(cfg.processed_dir, "meta.json"))`.

### 13.2 `scholargrid/artifacts.py`

```python
FACETS_PARQUET = "paper_facets.parquet"
PROBLEMS_PARQUET = "problems.parquet"
FINDINGS_PARQUET = "findings.parquet"
PROFILE_PARQUET = "paper_profile.parquet"
FACET_VOCAB_JSON = "facet_vocab.json"
PROBLEM_EMB_NPY = "problem_embeddings.npy"


def _parquet_optional(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
```

Add `Bundle` fields with defaults: `facets: pd.DataFrame`, `problems: pd.DataFrame`, `findings: pd.DataFrame`, `profiles: pd.DataFrame` (all `field(default_factory=pd.DataFrame)`), `facet_vocab: Dict = field(default_factory=dict)`, `problem_embeddings: Optional[np.ndarray] = None`. Load them at the end of `load_bundle`, and pass them as keyword args to `Bundle(...)` so old positional construction keeps working. Old bundles get empty frames.

### 13.3 Validation report (`scholargrid/validation.py`)

Add a section `extraction` to the report: papers with extraction, share LLM vs heuristic, mean verified problems per paper, `quote_verified_rate`, and the top 10 tasks/methods/datasets by count. Add an **advisory** gate `min_quote_verified_rate` (default 0.7, not required). A low rate means the prompt or the verifier needs work.

---

## 14. CLI: `python -m scholargrid.extract`

```text
python -m scholargrid.extract show 2401.01234         # pretty-print payload + verified problems with quotes
python -m scholargrid.extract sample --n 20 --out reports/extraction_sample.md   # for manual QA (plan 09)
python -m scholargrid.extract stats                    # modes, costs, verified rate, top facets
python -m scholargrid.extract rerun --ids 2401.01234 --mode llm   # force re-extraction (debug)
```

`sample` writes a Markdown file with, for each paper: title, link, extracted problems (statement + quote + section) and facets. It is the input for the extraction-quality review in plan 09.

---

## 15. Tests: `tests/test_extract.py`

1. **Schema clipping:** a 1,000-char `statement` is clipped, `specificity=7` → 3, and `code_url="github.com/x"` (no scheme) → `None`.
2. **Document builder:** markers `[S#]` appear, budgets are respected, `max_chars` truncates, and abstract-only documents get `[S0] ABSTRACT`.
3. **Heuristic on synthetic full text:** run `synthetic_sections` → build a document → `heuristic_extract_from_text` → ≥ 1 problem, every quote found verbatim by `verify_quotes`, and datasets include a `Synth*` name.
4. **Quote verification:**
   - exact quote → ratio 100,
   - quote with curly quotes and double spaces → verified, and the replaced quote equals the source span,
   - fabricated quote → dropped (`dropped == 1`).
5. **Canonicalization:**
   ```python
   counts = {"LLMs": 5, "large language models": 9, "Large Language Model": 2, "BERT": 4, "RoBERTa": 3}
   acr = {"llm": {"large language models": 7}}
   # rows follow list(counts): LLMs, large language models, Large Language Model, BERT, RoBERTa
   vecs = np.eye(5, dtype=np.float32)
   vecs[4] = 0.95 * vecs[3] + np.sqrt(1 - 0.95 ** 2) * np.eye(5, dtype=np.float32)[0]  # cos(BERT, RoBERTa) = 0.95
   m = canonicalize_values(counts, vecs, acr, 0.88, 80)
   assert m["LLMs"] == m["Large Language Model"] == "large language models"
   assert m["BERT"] != m["RoBERTa"]
   ```
6. **run_extract with FakeLLM:** first run → `llm == len(scope)`, model `fake`. Second run → `done_now == 0` (incremental). Changing `PROMPTS["extract_paper"].version` (monkeypatch) → everything reprocessed.
7. **Packing:** with `llm.tasks.extract.pack_size = 3` and a FakeLLM, every paper gets a record. With a stub router whose packed answer omits one item, that item is retried alone (count calls).
8. **Quota stop and resume:** a fake client whose `structured_many` returns `QuotaExhausted` for all but the first 2 items →
   - with `on_quota: heuristic`: `quota_stopped=True`, 2 `llm` records, the rest `heuristic`, and `pending_llm == n - 2`,
   - a second run with a working client upgrades exactly the heuristic rows to `llm`,
   - with `on_quota: skip`: the remaining papers have no rows,
   - with `stop`: the exception propagates after the 2 finished records are flushed.
   A paper whose LLM call keeps failing stops being retried after `max_llm_attempts`.
9. **Pipeline end-to-end** (`tests/test_pipeline.py`): after `run(cfg)`, `load_bundle(cfg)` has non-empty `problems` and `facets`, `facet_vocab["dataset"]` is non-empty, and `meta["extract"]["in_scope"] > 0`.

---

## 16. Acceptance checks

```bash
ruff check . && mypy && pytest -q
python -m scholargrid.pipeline --config configs/ci.yaml --no-publish
python -m scholargrid.extract stats --config configs/ci.yaml
# Real, free and laptop-light: ~20 papers = ~5 packed requests on the Gemini free tier
python -m scholargrid.llm estimate --task extract --n 20
python -m scholargrid.pipeline --config configs/domain_nlp_tiny.yaml --no-publish   # reuses plan 02's cached full text
python -m scholargrid.extract sample --n 10 --out reports/extraction_sample.md --config configs/domain_nlp_tiny.yaml
python -m scholargrid.llm usage --config configs/domain_nlp_tiny.yaml
python -m scholargrid.llm quota --config configs/domain_nlp_tiny.yaml
```

Scaling up to the 600-paper `domain_nlp_dev.yaml` scope happens **on GitHub**, not the laptop: run the
`dev-build` workflow (plan 01 §8.2) once a day until `extract stats` (add it as a workflow step) shows few
`pending_llm` papers.

**Manual QA before scaling up (required):** read `reports/extraction_sample.md` for 10 papers next to the papers themselves. Count:
(a) problems that are real author-stated problems,
(b) quotes that are exact,
(c) obvious missed problems.
Write the counts into `Docs/runbook.md`. Target: (a) ≥ 80% and (b) 100% (guaranteed by verification). If (a) is low, tighten the prompt and bump its `version` before running the full batch.

## 17. Pitfalls

- Bump `Prompt.version` on every prompt edit. Otherwise cached answers from the old prompt are reused.
- `wrap_untrusted("paper", ...)` must wrap the document on the **LLM** path. The heuristic strips the tags itself.
- Keep facet canonicalization **conservative**. A wrong merge creates fake "gaps" in plan 06; a missed merge only adds noise.
- Do not ship `extractions.parquet` (it contains full payloads) in the bundle. Only the exploded tables go there; the quotes in `problems.parquet` are ≤ 300 chars.
- On the free tier a large scope takes **days**. Let GitHub Actions run it daily (the `dev-build` workflow from plan 01 §8.2, later the scheduled workflow in plan 11), not the laptop. Track progress with `python -m scholargrid.extract stats` (papers pending an LLM pass) and `python -m scholargrid.llm quota`.
- If `LLMTruncated` appears often in `python -m scholargrid.llm usage`, lower `llm.tasks.extract.pack_size` or `extract.max_input_chars`, or raise `max_output_tokens`.
