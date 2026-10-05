# Software Requirements Specification (SRS)
## ScholarGrid
**Version:** 1.0  
**Document Type:** Software Requirements Specification  
**System:** ScholarGrid

---

## 1. Introduction

### 1.1 Purpose

The purpose of **ScholarGrid** is to provide an interactive, data-grounded system for exploring recent computer-science research.

The system transforms a corpus of recent arXiv computer-science papers into a semantic research landscape where users can:

- understand major research areas,
- search for research related to an interest or question,
- inspect representative papers,
- observe publication-growth patterns,
- investigate sparse or weakly connected research regions,
- and use the evidence to decide what to investigate further.

The system is intended as an **exploration and decision-support tool**, not an automatic research-gap or thesis-topic generator.

### 1.2 Problem Statement

Traditional paper-search tools help users find individual papers, but users may still struggle to understand:

- what major areas exist within a field,
- how those areas relate to one another,
- which areas are changing over time,
- what papers represent each area,
- and where potentially interesting connections or sparse regions exist.

The proposal therefore focuses on moving the user from **search → landscape understanding → trend analysis → evidence inspection → further investigation**. This is also reflected in the proposed user journey shown on page 3 of the proposal.

---

# 2. Scope

ScholarGrid will process recent arXiv computer-science papers and generate an interactive research landscape using:

**Data → Embeddings → Clustering → Labels → 2D Map → Growth Analysis → Semantic Search → Sparse-Neighborhood Analysis → Validation → Application → Optional Citation Analysis → Deployment.**

### 2.1 Core System Scope

The initial system shall provide:

1. arXiv research-data ingestion and preprocessing
2. Semantic embeddings of papers
3. Research-topic clustering
4. Interpretable cluster labels
5. Interactive 2D research landscape
6. Semantic research search
7. Publication-growth analysis
8. Sparse-neighborhood exploration
9. Evidence/relevant-paper inspection
10. Validation and robustness analysis
11. Methodology and limitation information
12. Public web deployment

### 2.2 Optional / Extended Scope

The system may additionally provide:

- semantic-citation missing-link analysis using OpenAlex,
- monthly refresh pipelines,
- beginner-oriented research entry recommendations.

The citation-based feature is considered a high-value extension rather than a mandatory MVP feature.

---

# 3. Target Users

The main target users are:

| User Type | Main Need |
|---|---|
| Students | Understand a new research area and explore thesis directions |
| Early researchers | Build a quick mental model of a field |
| Research teams | Observe research activity and related areas |
| ML engineers | Explore recent technical research landscapes |
| Technical recruiters | Understand active technical research directions |

The updated project plan specifically identifies students, early researchers, ML engineers, and technical recruiters as the primary audience.

---

# 4. Overall System Description

## 4.1 System Workflow

### Offline Analytical Pipeline

```text
arXiv Metadata
      ↓
Filter & Clean
      ↓
Title + Abstract Embeddings
      ↓
UMAP Dimensionality Reduction
      ↓
HDBSCAN Clustering
      ↓
Cluster Label Generation
      ↓
2D UMAP Visualization
      ↓
Temporal Growth Analysis
      ↓
Sparse-Neighborhood Analysis
      ↓
Optional Citation Analysis
      ↓
Validation
      ↓
Exported Artifacts
```

The expensive analytical computation is performed offline, while the deployed application primarily loads precomputed results and performs lightweight query embedding and nearest-neighbor retrieval.

---

# 5. Functional Requirements

## FR-01: Research Dataset Ingestion

The system shall:

- obtain a reproducible arXiv metadata dataset,
- retain computer-science (`cs.*`) papers,
- use a documented recent time window,
- retain at least:
  - arXiv ID,
  - title,
  - abstract,
  - category,
  - submission/update date,
- remove duplicate papers,
- remove papers with empty or very short abstracts.

The corpus snapshot and latest included date shall be recorded.

---

## FR-02: Paper Embedding Generation

The system shall:

- combine each paper's **title + abstract**,
- generate semantic embeddings,
- use the same embedding method for stored papers and user queries,
- save generated embeddings for reuse.

The initial baseline model shall be `all-MiniLM-L6-v2`.

---

## FR-03: Research Clustering

The system shall:

- reduce embeddings into approximately **5–10 dimensions using UMAP**,
- run **HDBSCAN** on the reduced representations,
- assign cluster identifiers to clustered papers,
- preserve papers classified as noise rather than forcing every paper into a cluster.

---

## FR-04: Cluster Label Generation

The system shall generate interpretable descriptions for research clusters.

Labels shall use:

- c-TF-IDF / BERTopic-style representative keywords,
- representative paper titles.

Representative evidence shall remain available so users can understand why a cluster received its label.

---

## FR-05: Interactive Research Landscape

The system shall generate a separate **2D UMAP** representation for visualization.

Users shall be able to:

- view papers on an interactive landscape,
- identify clusters,
- zoom and navigate the map,
- inspect individual papers,
- inspect paper title, category, date and cluster information.

The 2D map shall be treated as a visualization and not as the sole analytical basis for research conclusions.

---

## FR-06: Semantic Search

The system shall provide a search interface where users can enter a research topic, phrase or question.

The system shall:

1. embed the user's query,
2. compare the query with paper embeddings,
3. retrieve semantically nearest papers,
4. highlight those papers on the map,
5. display their cluster distribution,
6. provide representative paper information,
7. provide direct arXiv links.

If relevant papers belong to several clusters, the application shall expose that ambiguity rather than automatically assigning the query to one cluster.

---

# 6. Research Growth Requirements

## FR-07: Publication Growth Analysis

For each research cluster, the system shall calculate publication activity across time.

The baseline growth calculation shall be:

```text
raw_growth(c)
=
(recent_count(c) + 1)
/
(previous_count(c) + 1)
```

Growth shall then be normalized against publication growth across the complete computer-science corpus:

```text
relative_growth(c)
=
raw_growth(c)
/
raw_growth(all_CS)
```

A value greater than `1` indicates that the cluster's observed publication activity grew faster than the overall CS corpus during the chosen windows.

The application shall additionally display:

- monthly publication counts,
- selected growth window,
- stability information.

Growth should be evaluated across approximately:

- 3-month,
- 6-month,
- 12-month

windows.

---

# 7. Sparse-Neighborhood Exploration

## FR-08: Sparse-Neighborhood Detection

The system shall be able to identify candidate low-density regions located near relevant or high-growth research clusters.

The process shall include:

1. detecting candidate sparse regions on the 2D landscape,
2. checking those candidates in higher-dimensional embedding space,
3. retrieving nearby papers as supporting evidence,
4. repeating the analysis across different UMAP configurations,
5. rejecting candidates that exist only because of a particular 2D projection.

The system shall describe these areas as:

**"sparse neighborhoods"** or **"investigation leads."**

It shall not describe them automatically as proven research gaps.

---

# 8. Citation Missing-Link Analysis

## FR-09: Semantic-Citation Missing Links — Optional

If implemented, the system shall:

1. obtain citation relationships from OpenAlex or another documented source,
2. construct a semantic nearest-neighbor graph,
3. obtain citation relationships between papers or clusters,
4. identify research areas that are semantically close but have unexpectedly weak citation interaction,
5. exclude extremely small clusters,
6. expose representative papers from both sides.

Results shall be described as:

**"potential missing connections"**

rather than confirmed research gaps.

---

# 9. Evidence Requirements

## FR-10: Evidence Panel

For analytical signals displayed by the application, the user shall be able to inspect the underlying papers.

The evidence panel should show:

- paper title,
- relevant cluster,
- representative papers,
- paper metadata,
- direct arXiv link,
- supporting information behind a highlighted signal.

The project explicitly requires findings to remain traceable back to actual papers.

---

# 10. Filtering Requirements

## FR-11: Data Filtering

The application should allow users to filter the research landscape by:

- research category,
- date/time range.

Category and date filters are identified as a recommended application feature.

---

# 11. Cluster Details

## FR-12: Cluster Information Panel

When a cluster is selected, the application shall display relevant information such as:

- cluster label,
- representative keywords,
- representative papers,
- number of papers,
- monthly publication counts,
- relative growth,
- growth/stability information.

---

# 12. Methodology and Limitations

## FR-13: Methodology Information

The application shall provide an explanation of:

- embedding methodology,
- clustering methodology,
- dimensionality reduction,
- publication-growth calculation,
- sparse-neighborhood methodology,
- validation procedure.

---

## FR-14: Visible Limitations

Important methodological limitations shall be visible within the main application rather than existing only in documentation.

The application shall communicate that:

- visualization alone is not evidence,
- sparsity does not prove novelty,
- publication growth does not imply research quality,
- semantic similarity does not prove scientific compatibility,
- missing citations do not automatically mean missing knowledge.

---

# 13. Validation Requirements

## FR-15: Cluster Validation

The system shall evaluate cluster quality through:

- manual inspection of sample papers,
- representative-paper inspection,
- embedding cohesion where appropriate,
- robustness testing across parameter settings.

---

## FR-16: Search Validation

Semantic search shall be evaluated using approximately **10–20 known queries**.

The system should record a retrieval metric such as:

- Precision@k, or
- manually judged relevance rate.

---

## FR-17: Growth Validation

Growth calculations shall be repeated across:

- 3-month,
- 6-month,
- 12-month windows,
- alternate cutoff dates.

Highly unstable results should be identified.

---

## FR-18: Projection Robustness

Sparse-neighborhood candidates shall be tested using multiple:

- UMAP seeds,
- reasonable parameter configurations.

Candidates that disappear under reasonable alternative projections shall not be presented as robust findings.

---

# 14. Non-Functional Requirements

## NFR-01: Performance

The application should target a usable initial load time of approximately **10 seconds or less** on the selected free hosting platform.

The system should use WebGL-based visualization such as Plotly `scattergl` where necessary.

---

## NFR-02: Scalability

The analytical pipeline shall support a large scientific corpus.

When necessary:

- all papers may remain in the analytical dataset,
- only a representative subset may initially be rendered,
- detailed information may load on interaction.

---

## NFR-03: Reproducibility

The system shall:

- pin core dependency versions,
- store dataset snapshot information,
- store random seeds,
- record UMAP and HDBSCAN parameters,
- log paper counts after filtering,
- store intermediate embeddings,
- store cluster assignments,
- associate findings with pipeline versions where practical.

---

## NFR-04: Maintainability

The project shall use a structured repository separating:

```text
notebooks/
app/
data/processed/
configs/
reports/
```

The implementation plan defines this structure as part of the initial setup.

---

## NFR-05: Usability

The interface shall avoid presenting a large undifferentiated scatterplot.

Information should instead be organized through:

- searchable landscape,
- readable cluster labels,
- zoom/navigation,
- filters,
- cluster detail panels,
- evidence panels,
- growth visualizations.

---

## NFR-06: Traceability

Every important analytical signal should allow the user to inspect supporting research papers and understand how the result was calculated.

---

# 15. Data Requirements

### Primary Data Source

The primary corpus shall consist of recent **arXiv computer-science metadata and abstracts**.

### Required Fields

```text
arxiv_id
title
abstract
categories
submission/update_date
```

Optional:

```text
authors
citation information
```

### Derived Data

The pipeline shall generate and store information such as:

```text
embedding_vector
cluster_id
cluster_label
2D_UMAP_coordinates
publication_period
relative_growth
nearest_neighbors
sparse_signal
validation information
```

Citation-analysis data is only required if the extended missing-link functionality is implemented.

---

# 16. External Interface Requirements

## 16.1 User Interface

The web interface shall contain the following major components:

- Research Landscape
- Semantic Search Bar
- Cluster Details
- Growth Chart
- Sparse-Neighborhood Toggle
- Evidence Paper Panel
- Methodology Section
- Limitations Section
- Dataset Freshness Information

These elements are specifically included in the planned application design.

---

## 16.2 External Data Interfaces

The system shall interact with:

### arXiv

Used as the primary research-paper dataset.

### OpenAlex — Optional

Used for citation-aware missing-link analysis.

---

# 17. Deployment Requirements

The system shall precompute expensive analytical components such as:

- embeddings,
- clusters,
- UMAP coordinates,
- growth statistics.

The deployed application shall remain lightweight.

The proposed deployment target is **Hugging Face Spaces**, using Streamlit or Gradio for the MVP.

---

# 18. System Constraints

The following constraints apply:

- The 2D map must not be treated as scientific evidence by itself.
- Unsupervised clustering depends on model and parameter choices.
- Temporal sampling must not distort publication-growth measurements.
- Citation information may be incomplete or biased by paper age.
- Sparse areas may result from corpus limitations rather than genuine research underexploration.
- Manual cluster-label modification should be minimized and documented.

---

# 19. Functional Priority

| Priority | Requirement |
|---|---|
| **Must** | Reproducible corpus preprocessing |
| **Must** | Paper embeddings |
| **Must** | Research clustering |
| **Must** | Interpretable cluster labels |
| **Must** | 2D research landscape |
| **Must** | Semantic search |
| **Must** | Normalized growth analysis |
| **Must** | Representative papers |
| **Must** | Basic validation |
| **Should** | Robust sparse-neighborhood detection |
| **Should** | Category/date filtering |
| **Should** | Strong evidence UI |
| **High-value Stretch** | Semantic-citation missing-link analysis |
| **Stretch** | Monthly refresh |
| **Stretch** | Beginner research-entry recommendations |

This priority structure directly follows the project's MVP and stretch plan.

---

# 20. Acceptance Criteria

The system shall be considered complete for the planned first version when:

- the application is publicly accessible,
- the corpus snapshot and methodology are visible,
- semantic search performs sensibly against the documented benchmark,
- major clusters are interpretable,
- representative papers are available,
- publication growth is normalized and accompanied by absolute counts,
- at least one robustness/stability result is reported,
- sparse-neighborhood results pass both higher-dimensional and projection checks,
- citation-based outputs, if implemented, are identified as potential missing connections,
- the interface does not present sparsity as a proven research gap.

These conditions closely follow the project's defined **Definition of Done**.

---

# 21. Key System Boundary

ScholarGrid **does not determine**:

- whether a research topic is scientifically valuable,
- whether a topic is novel enough to publish,
- whether a topic is appropriate for a particular supervisor or laboratory,
- whether increased publication volume means research quality,
- whether two topics should be combined simply because they are close in embedding space.

Instead, the system assists the user in discovering and inspecting **evidence-backed areas worth further literature investigation**.

---

## 22. Final System Objective

The expected user journey is:

```text
User enters a research interest
            ↓
Semantic search finds relevant papers
            ↓
User sees where the interest lies on the research landscape
            ↓
User explores surrounding research clusters
            ↓
User examines publication trends
            ↓
System surfaces sparse / weakly connected investigation leads
            ↓
User inspects supporting papers
            ↓
User decides what area deserves deeper literature review
```

This captures the proposal's central idea: rather than simply searching paper-by-paper, the product helps the user develop a **larger view of the research landscape and then investigate evidence behind interesting signals**.

**SRS Version 1.0 — ScholarGrid**
