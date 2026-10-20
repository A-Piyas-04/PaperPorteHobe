"""Section-kind classification and gap-cue sentence detection (no ML; fast, transparent)."""
from __future__ import annotations

import re
from typing import List, Tuple

from .model import Cue, Section

_NUMBERING = re.compile(r"^\s*(?:(?:\d+|[A-Z])(?:\.\d+)*\.?|[IVXLC]+\.)\s+")

# First match wins. Order matters: "Conclusion and Future Work" -> future_work,
# "Limitations and Future Work" -> limitations.
_KIND_RULES: List[Tuple[str, re.Pattern]] = [
    ("references", re.compile(r"^(references|bibliography)$", re.I)),
    ("acknowledgements", re.compile(r"acknowledg", re.I)),
    ("limitations", re.compile(r"\blimitation", re.I)),
    ("threats_to_validity", re.compile(r"threats?\s+to\s+(the\s+)?validity", re.I)),
    ("future_work", re.compile(r"future\s+(work|direction|research)|open\s+(problem|question|challenge)s?|"
                               r"outlook|next\s+steps|research\s+agenda", re.I)),
    ("broader_impact", re.compile(r"broader\s+impact|societal\s+impact|impact\s+statement", re.I)),
    ("ethics", re.compile(r"ethic", re.I)),
    ("conclusion", re.compile(r"conclu|concluding|final\s+remarks|^summary$", re.I)),
    ("discussion", re.compile(r"discussion|lessons\s+learn", re.I)),
    ("appendix", re.compile(r"appendix|supplementa", re.I)),
    ("related_work", re.compile(r"related\s+work|prior\s+work|literature\s+review|previous\s+work", re.I)),
    ("background", re.compile(r"background|preliminar|problem\s+(formulation|statement|definition|setup)",
                              re.I)),
    ("introduction", re.compile(r"introduction|^overview$|^motivation$", re.I)),
    ("results", re.compile(r"\bresults?\b|\bfindings\b|\bevaluation\b|\banalysis\b|ablation", re.I)),
    ("experiments", re.compile(r"experiment|setup|implementation\s+details|empirical|benchmark|dataset",
                               re.I)),
    ("method", re.compile(r"method|approach|model|framework|architecture|proposed|algorithm|design",
                          re.I)),
]


def clean_heading(text: str) -> str:
    return _NUMBERING.sub("", " ".join(str(text).split())).strip(" .:")


def classify_heading(heading: str, parent_kind: str | None = None) -> str:
    h = clean_heading(heading)
    for kind, rx in _KIND_RULES:
        if rx.search(h):
            return kind
    return parent_kind or "other"


# --- sentences ---------------------------------------------------------------------------
_ABBREV = re.compile(r"\b(e\.g|i\.e|et al|cf|vs|Fig|Figs|Eq|Eqs|Sec|Tab|resp|approx|no)\.$", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def split_sentences(text: str) -> List[Tuple[int, str]]:
    """[(char_start, sentence)]. Joins splits that end in a common abbreviation."""
    out: List[Tuple[int, str]] = []
    pos, buf_start, buf = 0, 0, ""
    for part in _SPLIT.split(text):
        start = text.find(part, pos)
        pos = start + len(part)
        if buf and _ABBREV.search(buf):
            buf = buf + " " + part
            continue
        if buf:
            out.append((buf_start, buf.strip()))
        buf, buf_start = part, start
    if buf.strip():
        out.append((buf_start, buf.strip()))
    return out


# --- gap cues ------------------------------------------------------------------------------
CUE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("future_work", re.compile(r"\bfuture\s+(work|research|stud(y|ies)|direction)", re.I)),
    ("left_for_future", re.compile(r"\b(leave|left)\b[^.]{0,40}\b(for|to)\s+future", re.I)),
    ("we_plan", re.compile(r"\bwe\s+(plan|intend|hope)\s+to\b", re.I)),
    ("limitation", re.compile(r"\b(a|one|the|main|key|major|important|primary|current)\s+"
                              r"(main\s+|key\s+|major\s+)?limitations?\b", re.I)),
    ("limited_to", re.compile(r"\b(is|are|was|were|remains?)\s+limited\s+to\b", re.I)),
    ("cannot", re.compile(r"\b(our|this|the proposed)\s+(method|approach|model|system|work|framework)\s+"
                          r"(does\s+not|cannot|can\s+not|is\s+unable\s+to|fails?\s+to)\b", re.I)),
    ("open_question", re.compile(r"\b(remains?|is\s+still|are\s+still|stays?)\s+(an?\s+)?"
                                 r"(open|unclear|unsolved|unresolved|unexplored|elusive|challenging)", re.I)),
    ("open_problem", re.compile(r"\bopen\s+(problem|question|challenge|issue)s?\b", re.I)),
    ("not_explored", re.compile(r"\b(has|have)\s+not\s+(yet\s+)?been\s+(explored|studied|investigated|"
                                r"addressed|examined|evaluated)", re.I)),
    ("little_work", re.compile(r"\b(little|no|limited|scarce)\s+(prior\s+)?(work|attention|research|"
                               r"studies)\s+(has|have)\b", re.I)),
    ("out_of_scope", re.compile(r"\bbeyond\s+the\s+scope\b|\bout\s+of\s+(the\s+)?scope\b", re.I)),
    ("direction", re.compile(r"\b(interesting|promising|important|exciting|natural)\s+"
                             r"(direction|avenue|extension|next\s+step)", re.I)),
    ("does_not_generalize", re.compile(r"\b(may|might|does|do)\s+not\s+(generali[sz]e|transfer|scale|hold)\b",
                                       re.I)),
]
_CUE_KINDS = {"introduction", "discussion", "limitations", "future_work", "conclusion",
              "threats_to_validity", "broader_impact", "results", "abstract"}


def find_cues(sections: List[Section], max_sentence_chars: int = 600) -> List[Cue]:
    cues: List[Cue] = []
    for s in sections:
        if s.kind not in _CUE_KINDS or not s.text:
            continue
        whole = s.kind in {"limitations", "future_work", "threats_to_validity"}
        for start, sent in split_sentences(s.text):
            if len(sent) < 25 or len(sent) > max_sentence_chars:
                continue
            name = next((n for n, rx in CUE_PATTERNS if rx.search(sent)), None)
            if name is None and whole:
                name = f"in_{s.kind}"        # every sentence of a Limitations section is a candidate
            if name:
                cues.append(Cue(s.idx, s.kind, sent, name, start))
    return cues


_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    return _WS.sub(" ", text).strip()
