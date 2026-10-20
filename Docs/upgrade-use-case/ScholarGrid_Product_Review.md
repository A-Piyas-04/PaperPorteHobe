# ScholarGrid (PaperPorteHobe, `refactor` branch): Product Review from a Business Lead

**Reviewed:** 2026-10-19 · **Branch:** `refactor` (head `159776d`) · **Basis:** I read the source code, configs and all 14 docs in `Docs/`. I did not run the app with real users. The served-release numbers quoted below come from the project's own `SCHOLARGRID_REFACTORING_IMPLEMENTATION_PLAN.md` (release `2026.10.18.2`).

**The question:** Does this product help **early researchers** start research, **find research gaps** (the most important need), and **plan a path forward**?

---

## 1. Verdict

**No, not in its current form.** ScholarGrid is a well-engineered **semantic paper search plus a topic-cluster viewer**. Those are useful, but they are not what a beginner researcher is stuck on, and better-funded tools already do them.

- **Start researching:** partly. It finds papers and builds a reading list. It doesn't tell a beginner *what to read first*, *what the foundations are*, or *what the field's open questions are*.
- **Find research gaps:** **no.** The project explicitly refuses this job. The SRS says it is *"not an automatic research-gap or thesis-topic generator"* (§1.1, §21). The one feature near this job, "Investigation leads", is labelled *Experimental* and hidden under "More". Its algorithm finds **outlier papers**, not gaps (see §4.2).
- **Plan a path forward:** **no.** There is no learning path, no research question, no plan, no next experiment. The journey ends at "export BibTeX".

The core problem is strategic, not technical. **The SRS ranked "beginner research-entry recommendations" as the lowest priority (a "Stretch" item, §19), and it ruled gap-finding out of scope.** Those two things are the actual user need. Most of the engineering then went into infrastructure: versioned releases, rollbacks, ARI stability gates, bootstrap confidence intervals, DuckDB, FAISS, weekly OAI-PMH refresh, Sentry, Docker, and 14 planning documents. Meanwhile, by the project's own admission, there have been **no real-user usability sessions** (`ux-review.md`, "Boundaries").

The engineering is good and reusable (§6). The product needs to be pointed at a different problem.

---

## 2. Who the user is and what they need

An early researcher could be a final-year undergrad picking a thesis, a new MSc or PhD student, or an engineer moving into research. Their real questions, in the order they hit them:

| # | Beginner's question | What a good answer looks like |
|---|---|---|
| 1 | "What is this field, and what are its sub-problems?" | A small, readable map of sub-topics with plain-language summaries |
| 2 | "What should I read first?" | Surveys, then seminal papers, then current state of the art, in order |
| 3 | "What is still unsolved?" | **Concrete, evidence-backed open problems**, stated in one sentence |
| 4 | "Is this gap real, or has someone already done it?" | A novelty check against recent and live literature |
| 5 | "Can *I* do it?" | Feasibility: data, code, compute, skills needed |
| 6 | "How do I start?" | A research question, baselines, datasets, metrics, and a first experiment |
| 7 | "Who works on this?" | Labs, authors and venues (for supervisors, collaborators, submissions) |

ScholarGrid answers **#1 weakly**, **#2 not at all**, **#3 to #6 not at all**, and **#7 not at all**.

---

## 3. What the product actually does (from the code)

| Feature | Implementation | Value to a beginner |
|---|---|---|
| Search | Hybrid BM25 + dense embeddings (bge-small/MiniLM), acronym expansion, exact vs related sections, live arXiv/OpenAlex/Semantic Scholar | **Useful**, but Semantic Scholar and Google Scholar already do this with far more coverage |
| Research areas | UMAP → HDBSCAN over **48,015 papers sampled from all of arXiv CS (2023–2026)**, giving **77 areas** labelled by c-TF-IDF keywords | Low. Too coarse, labels are keyword lists, and **43% of papers are "noise"** in no area (noise fraction 0.43, per the refactoring plan) |
| Map | Sigma.js WebGL graph: areas, then papers, then neighbourhood | Looks good, but you can't learn much from it. The app itself says *"screen distance is only a rough similarity guide"* |
| Trends | Normalised growth with Poisson bootstrap CIs and stability gates | Statistically careful, but answers "is X growing?", which isn't a beginner's question (§4.5) |
| Investigation leads | Low-density points in 2D and high-dimensional space, robust across re-projections. **Max 12 for all of CS** | **Not gaps** (§4.2) |
| Paper detail | Title, authors, abstract excerpt, citations, similar papers, arXiv link | Fine, but standard |
| Reading list | Session-only, CSV/BibTeX export | Basic. **Lost on reload**, with no notes or tags |

---

## 4. The hard truths

### 4.1 The product disclaims its most important job
Gap-finding is the reason a beginner would choose this over Google Scholar. The project treats it as something to avoid. Every page repeats "not a proven gap", "quiet ≠ new", "connections do not establish a research gap". Honesty is good. But when the product has to keep warning users not to trust its signature feature, **that is evidence the feature is wrong, and more caveats won't fix it.**

### 4.2 "Investigation leads" detect outlier papers, not research gaps
From `scholargrid/sparse.py`:

1. Take papers in the bottom 20% of 2D local density that sit near any clustered paper.
2. Keep those whose distance to their 15th nearest neighbour in embedding space is above the median.
3. Keep those that stay low-density across 5 re-projections.
4. Output: one **anchor paper**, a robustness %, the nearest area name, and 3 neighbour titles.

A paper with few close neighbours in embedding space is usually one of these:
- an interdisciplinary or off-topic paper (e.g. a CS-category paper that is really physics or law),
- a paper with an unusual or poorly written abstract,
- a niche topic that is thin **in this sample** (the corpus is a stratified *sample* of arXiv CS, so sparsity partly measures the sampling),
- or work that lives mostly outside arXiv.

None of these is "an open problem". The lead card never says **what** is missing. "Lead 3 · 67% robust · Next to *Software, Agents & Code*" plus three titles gives a beginner nothing to act on. The leads are also **global**: 12 for all of computer science, unrelated to the user's search. Someone interested in medical image segmentation gets the same 12 leads as someone interested in compilers.

The SRS asked for sparse regions "near relevant or high-growth clusters" (FR-08). The code does not use growth or the user's query at all.

**Underlying issue:** gaps are a *semantic* property ("nobody has evaluated X under condition Y", "method M fails on Z and nobody has fixed it"). They are not a *geometric* property of an embedding space. You can't get them from density.

### 4.3 It reads only abstracts, and gaps live in the full text
Researchers state open problems in the **Limitations**, **Future Work** and **Discussion** sections. Abstracts are written to sell results and rarely mention what failed. The pipeline only embeds `title + abstract` (truncated to 900 characters in the UI). The richest gap signal in the literature never enters the system.

### 4.4 The granularity is wrong for beginners
77 areas across all of CS works out to roughly 600 papers per area, at the level of "LLMs, Agents & Reasoning". A beginner's thesis gap sits three or four levels lower, e.g. "KV-cache compression for long-context inference on mobile NPUs". There is no hierarchy and no way to zoom into sub-topics *of the user's own interest*. The labels are keyword strings (`pretty_label` joins the top c-TF-IDF terms), which beginners can't interpret, because they don't know the jargon yet.

### 4.5 Trends answer the wrong question, and can steer beginners the wrong way
"Area X grows 1.4× faster than CS" mostly means **"crowded and competitive"**, the hardest place for a beginner to contribute. The page never turns growth into advice ("hot and crowded, so you need a sharp niche" versus "steady with many unsolved problems"). The statistics are rigorous, but nothing tells the user what to do with them.

### 4.6 It can't show a beginner the foundations
The corpus window starts in **2023** (`config.yaml`) or **2021** (`production.yaml`). A beginner in NLP needs Attention Is All You Need (2017), BERT (2018) and similar papers. They aren't in the analysed corpus. Live search can surface them, but live papers are excluded from areas, the map and trends. There is no detection of surveys, seminal papers or tutorials, and no reading order. **"What do I read first?" goes unanswered.**

### 4.7 There is no path forward
After search and save, the product stops. There is no research question, hypothesis, baselines, datasets, evaluation metrics, milestone plan, or "who to talk to". That is a reading-list tool, not a research-starting tool.

### 4.8 Search quality is unproven
The search quality gate (`nDCG@10 ≥ 0.75`) is measured against **heuristic** labels: "expected category +1, expected keyword +1". `configs/search_judgments.json` was empty, and hand-grading is still marked **Open** (REF-04). The headline quality number is circular. Nobody knows whether search is actually good.

### 4.9 Effort went into the factory, not the product
| Built | Not built |
|---|---|
| Versioned releases, manifests, changelogs, rollback CLI | A single gap a user can act on |
| Bootstrap CIs, cutoff-stability CV, backtests, ARI gates | Any user interviews or usability tests |
| DuckDB store, FAISS HNSW, weekly OAI-PMH refresh, Sentry | Persistent reading lists or notes |
| Docker image, pre-commit, mypy, Playwright | A reading order or learning path |
| 14 docs, including 3 implementation plans and 2 SRS-style specs | A research plan output |

This is a common failure mode: excellent infrastructure for a product whose value hasn't been validated.

### 4.10 The competition is strong
Without a real gap-finding capability, the honest pitch is "a smaller Semantic Scholar with a cluster map". These tools already exist (feature sets change; check them yourselves):

- **Search and coverage:** Google Scholar, Semantic Scholar, OpenAlex (all papers, all years, all fields)
- **Visual related-paper maps:** Connected Papers, ResearchRabbit, Litmaps, Inciteful
- **AI literature assistants:** Elicit, Consensus, SciSpace, Undermind, and the "deep research" modes of general LLM assistants. These already produce summaries, comparison tables and "open questions" sections, though with variable grounding.

**The gap in the market is real:** none of these reliably give a beginner *evidence-backed, novelty-checked, feasibility-scored* research gaps plus a plan. That is the opening this project should go after.

---

## 5. What is good (keep it)

To be fair, a lot here is reusable:

- **The offline/online split.** Heavy computation runs offline into a versioned bundle, and the app stays light. This is exactly the right base for precomputing expensive LLM extractions.
- **Hybrid search** (BM25 + dense + acronym expansion + exact/related sections). This becomes the retrieval layer for the gap engine.
- **OpenAlex enrichment** (citations, references, coupling edges). This becomes the basis for seminal-paper detection and citation missing links.
- **Statistical honesty.** Gates, CIs and "counts only" fallbacks show a team that cares about not over-claiming. Keep that culture and apply it to gap evidence.
- **Evidence traceability** (NFR-06). Every signal links to papers. A gap engine needs exactly this.
- **Clean UX basics:** a clear landing page, a skippable tour, BibTeX export, responsive layout.

---

## 6. How to fix it: the redesigned product

### 6.1 New positioning
> **ScholarGrid takes an early researcher from a vague interest to a defensible research question in one sitting, with every claim backed by papers.**

### 6.2 New user journey
```
1. ORIENT     "I'm interested in X"        → sub-topic map *of X*, plain-language summaries
2. FOUNDATIONS "What do I read first?"     → ordered path: survey → seminal → SOTA → frontier
3. FRONTIER   "What is unsolved?"          → Gap Cards (evidence, type, novelty check)
4. CHOOSE     "Which gap fits me?"         → filter by feasibility, skills, compute, data
5. PLAN       "How do I start?"            → research question, baselines, datasets, 8–12 week plan
6. KEEP       "Save my work"               → persistent project with notes and exports
```

### 6.3 The core feature: a real Gap Engine (replace density-based leads)
Every gap should be a **Gap Card**:

```
GAP: "No existing work evaluates speculative decoding on ARM/mobile NPUs
      with quantised draft models."
Type:            Combinatorial (method × setting)
Evidence:        11 papers on speculative decoding (all GPU/server), 7 papers on
                 mobile NPU inference (none use speculative decoding)
Stated by authors: 3 papers mention "edge deployment" as future work (quoted, linked)
Novelty check:   Live search on arXiv + Semantic Scholar, last 6 months: 1 partial match (linked)
Why it may NOT be a gap: hardware access; may be in industry blogs, not papers
Feasibility:     Medium · needs: PyTorch, an Android device, open 1–7B models
Suggested first step: reproduce paper A's baseline on device B
```

The engine should detect several **gap types**, each with its own method:

| Gap type | How to detect it | Beginner value |
|---|---|---|
| **Stated open problems** | Extract Limitations / Future Work sentences from full text (arXiv HTML/LaTeX, or open-access PDFs via Semantic Scholar/OpenAlex). Cluster similar statements across papers. Count how many papers raise each one, and whether later papers claim to solve it. | **Very high.** The authors themselves say it's open |
| **Combinatorial gaps** | Use an LLM to extract structured facets per paper (task, method, dataset, domain/language, modality, metric). Build matrices such as method × task and task × domain. Look for empty or thin cells next to dense rows and columns. Confirm absence with a live search. | **Very high.** The classic beginner gap ("apply M to T/D") |
| **Evaluation gaps** | Methods evaluated on only one benchmark, no real-world or low-resource evaluation, no robustness or fairness evaluation | High. Feasible for beginners |
| **Contradictions** | Papers reporting conflicting results on the same task and dataset | High. Good for replication studies |
| **Citation missing links** (FR-09, already planned) | Semantically close communities that rarely cite each other | Medium. Bridge ideas |
| **Abandoned but important** | Problems that were highly cited, show declining activity, and still have stated open issues | Medium |

**Guardrails** (carry over the team's honesty culture):
- every claim links to a **quoted sentence** from a paper; no claim without a source,
- an automatic **novelty check** against recent and live papers before showing a card,
- an explicit **"why this might not be a gap"** line on every card,
- a confidence level based on how many independent papers support it.

### 6.4 Make everything query-scoped and fine-grained
- On a query, build a **topic workspace**: retrieve 500–2,000 papers (corpus + live + one hop of citation expansion through OpenAlex/Semantic Scholar), then cluster *those* into 8–20 sub-topics.
- Label sub-topics with an LLM: a short name plus a 2-sentence plain-English summary, with the keywords as fallback.
- Support hierarchy: area → sub-topic → micro-topic.
- Keep the global 77-area map only as a secondary "browse" view.

### 6.5 Foundations and learning path
- Remove the 2023 floor for the **citation backbone**: keep recent papers for analysis, but pull referenced older papers into the workspace.
- **Seminal papers** = most co-cited references within the topic workspace.
- **Surveys** = title/abstract classification ("survey", "review", "overview", "tutorial", "SoK").
- An ordered reading path with a reason for each paper ("defines the problem", "current SOTA", "best recent survey"), plus a prerequisite-concepts list.

### 6.6 From gap to plan
For a chosen gap, generate a **starter research plan** grounded in the evidence papers:
- research question and hypothesis,
- 2–3 baselines to reproduce (with code links where they exist),
- datasets and metrics used by the closest work,
- an 8–12 week milestone plan and main risks,
- target venues, and labs/authors active on this (for supervisor and collaborator search).

### 6.7 Persistence
Use accounts, or at least a saved project (link or file): reading list, per-paper notes, tags ("gap evidence", "baseline", "must read"), and saved gap cards. Session-only storage kills the product for any task longer than an hour, and starting research takes weeks.

### 6.8 Translate trends into advice
Show a beginner-facing quadrant: **growth × crowdedness × number of open gaps**.
- "Growing + crowded" → competitive; you need a sharp niche
- "Growing + few papers + many stated gaps" → **emerging opportunity**
- "Stable + many unresolved stated problems" → steady, good thesis territory

---

## 7. Narrow the scope to win

Covering *all of CS* at a 10%-style sample is why everything is shallow. **Pick one sub-field you know well** (e.g. `cs.CL` NLP, or `cs.CV` medical imaging), go deep there, prove the value, then expand. Depth beats breadth for the first 100 users, and it makes LLM extraction costs affordable on a free tier.

---

## 8. What to pause

- More infrastructure work: release/rollback polish, FAISS, weekly refresh hardening, Sentry, DuckDB pruning.
- Map polish (sphere nodes, hover effects; the last two commits). The map isn't where the user's value lies.
- New planning documents. There are already three implementation plans and two specs. The next document should be **interview notes**.
- Global sparse leads. Hide them once the Gap Engine ships.

---

## 9. How to know whether it works (validation plan)

### Step 0: talk to users (week 1–2, before writing code)
- Interview **8–10 early researchers** (thesis undergrads, first-year MSc/PhD). Ask: how did you find your topic? What took longest? What did your supervisor reject, and why?
- **Wizard-of-Oz test:** for 3 topics, hand-make 5 Gap Cards each (using an LLM plus manual checking), show them to students and 2–3 supervisors, and record which they'd pursue. If hand-made cards don't excite them, automated ones won't either.

### Metrics
| Metric | Target for first version |
|---|---|
| **North star:** % of users who leave with a research question a supervisor rates "viable" | ≥ 40% |
| Gap precision: % of gap cards a domain expert rates "real and open" | ≥ 60% |
| Novelty false-positive rate: gaps shown that are already solved | ≤ 15% |
| Time to a shortlist of 3 candidate gaps | < 60 min (vs. a Google Scholar + ChatGPT baseline) |
| Search quality: nDCG@10 against **human** judgments (finish REF-04) | Set a baseline, then improve |
| Return within 7 days | ≥ 30% |

### Benchmark test
Give 2 groups of students the same topic. Group A uses Google Scholar + a general LLM; group B uses ScholarGrid. Supervisors blind-rate the gaps and plans each group produces. **If ScholarGrid doesn't clearly win this test, it has no reason to exist.**

---

## 10. Roadmap

| Phase | Weeks | Deliverable |
|---|---|---|
| **0. Discovery** | 1–2 | Interviews, Wizard-of-Oz gap cards, pick one sub-field |
| **1. Topic workspace** | 3–5 | Query-scoped retrieval with citation expansion, fine-grained sub-topics with LLM labels and summaries, reading path (surveys → seminal → SOTA) |
| **2. Gap Engine v1** | 5–9 | Offline full-text "limitations/future work" extraction and facet extraction for the chosen sub-field; stated-gap and combinatorial-gap cards; live novelty check |
| **3. Plan + persistence** | 9–11 | Gap-to-plan generator; saved projects with notes and tags |
| **4. Evaluation** | 11–12 | Benchmark test vs. baseline, expert rating of gap cards, then decide go / iterate / pivot |

**Technical fit:** LLM extraction runs **offline** in the existing pipeline (new `extract_full_text` and `facets` stages), and results ship in the bundle. The current architecture already supports this. Only the query-time novelty check and plan generation need online LLM calls, which can be rate-limited or bring-your-own-key on a free tier.

**Risks:**
- **LLM hallucination.** Allow no claim without a quoted source sentence, show sources inline, and use extraction rather than generation wherever possible.
- **Full-text access and licensing.** arXiv HTML/LaTeX is broadly available, but check licences per paper. Fall back to the abstract when the full text isn't available.
- **Cost.** Precompute offline, cache per topic, limit to one sub-field at first.
- **Users trust gaps too much.** Keep the "why this might not be a gap" line and the novelty check visible. This is the team's existing strength.

---

## 11. Summary

| Question | Answer |
|---|---|
| Is it helpful to intended users? | **Slightly.** A decent search with nice visuals, but not better than free tools beginners already use. |
| Does it solve a problem? | It solves "see clusters of CS papers", which **isn't the beginner's problem**. |
| Does it help find research gaps? | **No.** It rules gap-finding out of scope, and its "leads" are outlier papers. |
| Does it help plan a path? | **No.** No reading order, no research question, no plan. |
| Is the project worth continuing? | **Yes, if it pivots.** The engineering base is solid and the market gap (evidence-backed, novelty-checked gaps plus a starter plan for beginners) is real and under-served. |
| Most important next action | **Talk to 10 beginner researchers and hand-test Gap Cards before writing more code.** |
