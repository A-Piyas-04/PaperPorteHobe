"""No-LLM extraction from the same [S#]-marked document the LLM would see.

Used when no LLM is available, when the daily free quota is used up (filled in now, upgraded on a later
run), by FakeLLM in tests, and for abstract-only papers in ``mode: heuristic``. Deliberately simple,
transparent and conservative.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

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
_ACRONYM = re.compile(r"(?<![.\w])([A-Z][A-Za-z]*[A-Z][A-Za-z0-9\-]*)\b")   # not "cs.CL"
_STOP_ACR = {"AI", "ML", "NLP", "CV", "GPU", "CPU", "API", "URL", "SOTA", "TITLE", "ABSTRACT", "GAP", "CUE",
             "IID", "OOD", "US", "UK", "EU", "CIT", "MATH", "SENTENCES", "DETECTED", "AUTOMATICALLY"}
_METRIC_RX = {m: re.compile(rf"(?<![a-z]){re.escape(m)}(?![a-z])") for m in _METRICS}
_TYPE_BY_CUE = {"future_work": "future_work", "left_for_future": "future_work", "we_plan": "future_work",
                "direction": "future_work", "open_question": "open_challenge_in_field",
                "open_problem": "open_challenge_in_field", "not_explored": "unexplored_setting",
                "little_work": "unexplored_setting"}
_PROBLEM_KINDS = {"limitations", "future_work", "discussion", "conclusion", "threats_to_validity"}


def _group(rx: str, text: str, flags: int = 0, default: str = "") -> str:
    m = re.search(rx, text, flags)
    return m.group(1) if m else default


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


def _compute(low: str) -> str:
    if re.search(r"\b\d+\s*(x\s*)?(a100|h100|v100|gpus)\b", low):
        return "multi_gpu"
    if re.search(r"\b(a100|h100|v100|rtx|gpu)\b", low):
        return "single_gpu"
    return "unknown"


def heuristic_extract_from_text(document: str) -> PaperExtraction:
    secs = _sections(document)
    title = _group(r"^TITLE:\s*(.+)$", document, re.M).strip()
    subject = title.split(":")[0].strip() or "the approach"
    full = " ".join(b for *_, b in secs)
    low = full.lower()

    problems: List[StatedProblem] = []
    for label, kind, _, body in secs:
        if kind not in _PROBLEM_KINDS:
            continue
        for _, sent in split_sentences(body):
            cue: Optional[str] = next((n for n, rx in CUE_PATTERNS if rx.search(sent)), None)
            if cue is None and kind not in {"limitations", "future_work"}:
                continue
            ptype = _TYPE_BY_CUE.get(cue or "", "limitation_of_this_work" if kind == "limitations" else "future_work")
            statement = re.sub(r"\b(our|this)\s+(approach|method|model|work|framework)\b", subject, sent, flags=re.I)
            problems.append(StatedProblem(statement=statement, quote=sent[:300], section=label,
                                          problem_type=ptype, specificity=2))  # type: ignore[arg-type]
    datasets = {(m.group(1) or m.group(2)) for m in _DATASET.finditer(full) if (m.group(1) or m.group(2))}
    for m in _ON_DATASETS.finditer(full):
        datasets |= {d.strip(" ,.") for d in re.split(r",|\band\b", m.group(1)) if d.strip(" ,.").startswith("Synth")}
    methods = [MethodMention(name=subject, role="proposed")] if ":" in title else []
    methods += [MethodMention(name=a, role="used")
                for a in sorted(set(_ACRONYM.findall(full)) - _STOP_ACR - datasets)[:8]]
    task = _group(r"\bfor\s+([a-z][a-z \-]{3,40})$", title.lower())
    first = split_sentences(full)
    return PaperExtraction(
        one_line_summary=(first[0][1] if first else title)[:240],
        tasks=[task] if task else [],
        methods=methods,
        datasets=sorted(datasets),
        metrics=[m for m, rx in _METRIC_RX.items() if rx.search(low)],
        domains=["general"],
        languages=[lang for lang in _LANGS if re.search(rf"\b{lang}\b", full)],
        modalities=[k for k, rx in _MODALITY.items() if re.search(rx, low)][:3] or ["other"],  # type: ignore[arg-type]
        contribution_types=["survey"] if re.search(r"\b(survey|review)\b", title.lower()) else ["new_method"],
        evaluation=EvalProfile(
            real_world=True if re.search(r"real[- ]world|in the wild|deployed", low) else None,
            human_eval=True if re.search(r"human evaluation|user study|annotators", low) else None,
            robustness=True if re.search(r"robust|adversarial|distribution shift|out-of-distribution", low) else None,
            efficiency_reported=True if re.search(r"latency|throughput|memory footprint|flops", low) else None,
            low_resource=True if "low-resource" in low or "low resource" in low else None,
            multilingual=True if re.search(r"multilingual|cross-lingual", low) else None),
        open_problems=problems,
        findings=[],
        code_url=_group(r"(https?://github\.com/[\w\-./]+)", full) or None,
        data_released=None,
        compute=_compute(low),  # type: ignore[arg-type]
    )
