"""Guards between the LLM and the user: no invented entities, no unknown evidence ids, no overclaiming."""
from __future__ import annotations

import re
from typing import Any, List, Set

from .model import GapCandidate

_ENTITY = re.compile(r"\b(?:[A-Z][a-z]*[A-Z][A-Za-z0-9\-]*|[A-Z]{2,}[A-Za-z0-9\-]*|[A-Z][a-z]+-\d+[A-Za-z]*)\b")
_ALLOWED = {"AI", "ML", "NLP", "LLM", "LLMs", "GPU", "GPUs", "CPU", "API", "OOD", "SOTA", "E1", "E2", "E3",
            "E4", "E5", "E6", "E7", "E8", "US", "UK", "EU"}
_OVERCLAIM = re.compile(r"\b(no one|nobody) has ever\b|\bfirst ever\b", re.I)


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


def verify_text(draft: Any, c: GapCandidate) -> List[str]:
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
    if _OVERCLAIM.search(draft.statement):
        issues.append("overclaiming language")
    return issues
