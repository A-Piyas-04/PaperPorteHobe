"""Prompts for per-paper extraction.

The caller wraps the document (``document=wrap_untrusted("paper", doc.text)``), not the template, so
the fake handler and the heuristic see the same text; the heuristic strips the tags itself.
"""
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
