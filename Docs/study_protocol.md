# ScholarGrid study protocol

ScholarGrid promises *real, open, feasible* research gaps. That is an empirical claim. This document is the
plan for the non-code validation that backs it, alongside the rating rounds in `scholargrid/evaluation/`
(Evaluate page, `python -m scholargrid.evaluation`).

## 1. Discovery interviews

**When:** before and during building plans 01–03.
**Who:** 8–10 early researchers: final-year undergraduates, first-year master's/PhD students, and engineers
moving into research.
**Format:** 30 minutes, semi-structured, recorded as anonymized notes (no names, no institution).

Interview guide:

1. How did you find your current or last research topic?
2. What took longest?
3. What did a supervisor reject, and why?
4. Which tools did you use (Google Scholar, Semantic Scholar, ChatGPT/Claude, Connected Papers, …), and where
   did they fail?
5. What would make you trust a suggested research gap?

**Synthesis:** cluster the notes into needs and rank the top 5 by how many participants mentioned them.
Record the ranking in `Docs/runbook.md`.

## 2. Wizard-of-Oz gap cards

For 3 topics, hand-make 5 cards each using the Gap Card template (statement, evidence quotes, "why it
might not be a gap", first steps). Show them to students and to 2–3 supervisors.

Measure which cards they would pursue, which they would reject, and why.

**Go/no-go for the gap engine (plan 06):** at least 50% of the hand-made cards are rated "worth pursuing" by
supervisors. If not, rethink before building the engine.

## 3. Benchmark study (after plan 08)

| Item | Plan |
|---|---|
| Design | Within-subject, or two groups of at least 6 students each, on the same 2 topics |
| Arm A | Google Scholar + a general-purpose LLM assistant, 60 minutes |
| Arm B | ScholarGrid, 60 minutes |
| Output | A shortlist of 3 candidate gaps plus a one-paragraph plan |
| Rating | Blind rating by 2+ supervisors with the gap rubric (Evaluate page) plus "Would you supervise this?" |
| Primary outcome | Share of shortlisted gaps rated real **and** open **and** feasible |
| Secondary outcomes | Time to shortlist; self-rated confidence; number of verifiable citations; errors (claims not supported by any paper) |
| Analysis | Compare proportions with a two-sided test, or a Bayesian beta-binomial estimate given the small samples. Report effect sizes and intervals, not only p-values |
| Decision | ScholarGrid must clearly beat arm A on the primary outcome to justify continued investment |

Gap rubric (the same as the Evaluate page): real research problem (yes/partly/no); still open (yes/unsure/no);
specific enough to start (1–3); feasible for a master's student in about 3 months (yes/maybe/no); evidence
supports the claim (yes/partly/no); would you suggest it to a student (yes/no).

## 4. Ethics and consent

- Give participants an information sheet: purpose, what is recorded, how long it is kept.
- Participation is voluntary, and participants may withdraw at any time without giving a reason.
- No personal data goes into product telemetry. Study notes are anonymized.
- Check whether your university requires IRB/ethics approval for user studies **before** recruiting.

## 5. Templates

**Consent form (short):**

> I have read the information sheet for the ScholarGrid study. I understand that taking part is voluntary,
> that I can withdraw at any time, and that my answers will be stored without my name. I agree to take part.
> Signature / date.

**Interview guide:** section 1 above.

**Rating sheet:** mirror the Evaluate page rubric (section 3), one row per card:

| Card | Real? | Open? | Specific (1–3) | Feasible? | Evidence supports? | Suggest? | Would supervise? | Comment |
|---|---|---|---|---|---|---|---|---|

**Results table:**

| Arm | Participants | Shortlisted gaps | Real & open & feasible | Share (95% interval) | Median time to shortlist | Unsupported claims |
|---|---|---|---|---|---|---|
| A: Scholar + LLM | | | | | | |
| B: ScholarGrid | | | | | | |
