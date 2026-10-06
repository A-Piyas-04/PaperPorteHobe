"""Okapi BM25 lexical index over titles + abstracts.

Complements dense retrieval for exact-term queries (model names, acronyms,
dataset names). Stored in the bundle as a sparse term-frequency matrix plus
vocabulary, so the app needs no extra dependency.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from .utils import ensure_dir, load_json, save_json

TOKEN_PATTERN = r"(?u)\b[\w][\w\-\.]*[\w]\b|\b\w\b"
_TOKEN_RE = re.compile(TOKEN_PATTERN)


def tokenize(text: str) -> List[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class BM25Index:
    def __init__(self, tf: sp.csc_matrix, vocab: Dict[str, int], k1: float = 1.5, b: float = 0.75):
        self.tf = tf.tocsc()
        self.vocab = vocab
        self.k1, self.b = k1, b
        n_docs = tf.shape[0]
        self.doc_len = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(self.doc_len.mean()) if n_docs else 1.0
        df = np.diff(self.tf.indptr)
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    @classmethod
    def build(cls, texts: List[str], max_features: Optional[int] = 200000) -> "BM25Index":
        from sklearn.feature_extraction.text import CountVectorizer

        vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                              max_features=max_features, dtype=np.float32)
        tf = vec.fit_transform(texts)
        vocab = {str(k): int(v) for k, v in vec.vocabulary_.items()}
        return cls(tf.tocsc(), vocab)

    def query_terms(self, query: str) -> List[str]:
        return [t for t in dict.fromkeys(tokenize(query)) if t in self.vocab]

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.tf.shape[0], dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in self.query_terms(query):
            j = self.vocab[term]
            start, end = self.tf.indptr[j], self.tf.indptr[j + 1]
            rows = self.tf.indices[start:end]
            f = self.tf.data[start:end]
            out[rows] += self.idf[j] * f * (self.k1 + 1) / (f + norm[rows])
        return out

    def topk(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        nz = np.flatnonzero(s)
        if len(nz) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(k, len(nz))
        part = nz[np.argpartition(-s[nz], k - 1)[:k]]
        order = part[np.argsort(-s[part])]
        return order.astype(np.int64), s[order]

    # -- persistence -----------------------------------------------------------
    def save(self, directory: str) -> None:
        ensure_dir(directory)
        sp.save_npz(os.path.join(directory, "bm25_tf.npz"), self.tf.tocsr())
        save_json({"vocab": self.vocab, "k1": self.k1, "b": self.b},
                  os.path.join(directory, "bm25_meta.json"))

    @classmethod
    def load(cls, directory: str) -> Optional["BM25Index"]:
        tf_path = os.path.join(directory, "bm25_tf.npz")
        meta_path = os.path.join(directory, "bm25_meta.json")
        if not (os.path.exists(tf_path) and os.path.exists(meta_path)):
            return None
        meta = load_json(meta_path)
        return cls(sp.load_npz(tf_path).tocsc(), meta["vocab"], meta["k1"], meta["b"])


def reciprocal_rank_fusion(rankings: List[np.ndarray], k: int = 60) -> Dict[int, float]:
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (k + rank)
    return fused
