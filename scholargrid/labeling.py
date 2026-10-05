"""FR-04  Cluster label generation.

Interpretable descriptions for each cluster using class-based TF-IDF
(c-TF-IDF, the BERTopic formulation) over cluster-aggregated documents, plus
representative papers nearest each cluster's embedding centroid. Representative
evidence is retained so users can see *why* a cluster received its label.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger

log = get_logger("label")


def generate_labels(df: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray,
                    cfg: Config) -> Dict[int, Dict]:
    from sklearn.feature_extraction.text import CountVectorizer

    top_k = int(cfg["labeling"]["top_keywords"])
    n_rep = int(cfg["labeling"]["n_representative"])
    cluster_ids = sorted({int(c) for c in labels if c != -1})
    if not cluster_ids:
        return {}

    texts = (df["title"].astype(str) + ". " + df["abstract"].astype(str)).tolist()
    # One aggregated document per cluster for c-TF-IDF.
    docs_per_cluster, index_per_cluster = {}, {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        index_per_cluster[cid] = idx
        docs_per_cluster[cid] = " ".join(texts[i] for i in idx)

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=1, max_features=10000)
    counts = vectorizer.fit_transform([docs_per_cluster[c] for c in cluster_ids])
    terms = np.array(vectorizer.get_feature_names_out())
    ctfidf = _c_tf_idf(counts.toarray().astype(float))

    out: Dict[int, Dict] = {}
    for row, cid in enumerate(cluster_ids):
        order = np.argsort(-ctfidf[row])[:top_k]
        keywords = [str(terms[j]) for j in order if ctfidf[row, j] > 0]
        idx = index_per_cluster[cid]
        rep_idx = _representative_indices(embeddings, idx, n_rep)
        rep = df.iloc[rep_idx]
        out[cid] = {
            "cluster_id": cid,
            "size": int(len(idx)),
            "keywords": keywords,
            "label": _label_from_keywords(keywords),
            "representative_papers": [
                {"arxiv_id": r.arxiv_id, "title": r.title,
                 "primary_category": r.primary_category,
                 "date": r.date.strftime("%Y-%m-%d")}
                for r in rep.itertuples()
            ],
            "top_categories": df.iloc[idx]["primary_category"].value_counts().head(5).to_dict(),
        }
    log.info("Generated labels for %d clusters.", len(out))
    return out


def _c_tf_idf(tf: np.ndarray) -> np.ndarray:
    """BERTopic c-TF-IDF: X = tf * log(1 + A / df), A = mean words per class."""
    words_per_class = tf.sum(axis=1)
    A = words_per_class.mean()
    df_term = (tf > 0).sum(axis=0)
    idf = np.log(1.0 + A / np.maximum(df_term, 1e-9))
    tf_norm = tf / np.maximum(words_per_class[:, None], 1e-9)
    return tf_norm * idf


def _representative_indices(embeddings: np.ndarray, idx: np.ndarray, n: int) -> List[int]:
    centroid = embeddings[idx].mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-9)
    sims = embeddings[idx] @ centroid
    order = np.argsort(-sims)[: min(n, len(idx))]
    return [int(idx[o]) for o in order]


def _label_from_keywords(keywords: List[str]) -> str:
    return pretty_label(keywords)


_GENERIC = {
    "based", "using", "model", "models", "learning", "data", "approach",
    "method", "methods", "paper", "results", "propose", "proposed", "new",
    "task", "tasks", "framework", "system", "systems", "study", "performance",
}
_ACRONYMS = {
    "ai": "AI", "llm": "LLM", "llms": "LLMs", "gpu": "GPU", "gpus": "GPUs",
    "kv": "KV", "moe": "MoE", "eeg": "EEG", "rl": "RL", "fl": "FL", "3d": "3D",
    "2d": "2D", "gnn": "GNN", "gnns": "GNNs", "epr": "EPR", "nlp": "NLP",
    "iot": "IoT", "api": "API", "ml": "ML", "vlm": "VLM", "vlms": "VLMs",
    "vla": "VLA", "rag": "RAG", "cuda": "CUDA", "sql": "SQL", "qa": "QA",
    "mri": "MRI", "ct": "CT", "asr": "ASR", "tts": "TTS", "hpc": "HPC",
}


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _title(term: str) -> str:
    return " ".join(_ACRONYMS.get(w, w.capitalize()) for w in term.split())


def pretty_label(keywords: List[str], max_terms: int = 3) -> str:
    """Human-readable cluster name from ranked c-TF-IDF keywords.

    Drops near-duplicates (agent/agents), lets a bigram replace the unigrams
    it contains (kv + cache -> KV Cache), skips generic words when possible,
    and title-cases the result: ``"Software, Agents & Code"``.
    """
    if not keywords:
        return "Unlabeled area"

    def pick(skip_generic: bool) -> List[str]:
        chosen: List[tuple] = []
        for kw in keywords:
            kw = str(kw).strip().lower()
            words = kw.split()
            if not words or (skip_generic and all(w in _GENERIC for w in words)):
                continue
            stems = {_stem(w) for w in words}
            if len(words) > 1:
                chosen = [c for c in chosen if not c[1] <= stems]
            covered = set().union(*[c[1] for c in chosen]) if chosen else set()
            if stems <= covered:
                continue
            if len(words) == 1 and any(
                    len(c) >= 4 and (words[0].startswith(c) or c.startswith(words[0]))
                    for c in covered):
                continue
            chosen.append((kw, stems))
            if len(chosen) >= max_terms:
                break
        return [c[0] for c in chosen]

    terms = pick(skip_generic=True)
    if len(terms) < 2:
        terms = pick(skip_generic=False)
    terms = [_title(t) for t in terms]
    if len(terms) == 1:
        return terms[0]
    return ", ".join(terms[:-1]) + " & " + terms[-1]
