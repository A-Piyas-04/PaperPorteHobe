"""FR-02  Paper embedding generation.

A single :class:`Embedder` encodes both stored papers and user queries with the
*same* method (a hard FR-02 requirement). Two backends:

* ``minilm`` - sentence-transformers ``all-MiniLM-L6-v2`` (SRS baseline).
* ``tfidf``  - TfidfVectorizer -> TruncatedSVD -> L2 normalise. A fully
  reproducible, offline fallback that needs only scikit-learn.

The fitted state is saved to disk so the deployed app embeds queries with the
identical transformation without recomputing corpus embeddings.
"""
from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .utils import ensure_dir, get_logger, l2_normalize

log = get_logger("embed")


def paper_text(df: pd.DataFrame) -> List[str]:
    """FR-02: combine title + abstract."""
    return (df["title"].astype(str) + ". " + df["abstract"].astype(str)).tolist()


class Embedder:
    def __init__(self, backend: str, cfg: Config):
        self.backend = backend
        self.cfg = cfg
        self._model = None          # minilm
        self._vectorizer = None     # tfidf
        self._svd = None            # tfidf
        self.dim: int | None = None

    # -- fit / encode -------------------------------------------------------
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if self.backend == "minilm":
            return self._fit_minilm(texts)
        if self.backend == "tfidf":
            return self._fit_tfidf(texts)
        raise ValueError(f"Unknown embedding backend: {self.backend}")

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.backend == "minilm":
            vecs = self._model.encode(
                texts, batch_size=self.cfg["embedding"]["batch_size"],
                show_progress_bar=False, normalize_embeddings=False,
            )
            return l2_normalize(np.asarray(vecs, dtype=np.float32))
        if self.backend == "tfidf":
            x = self._vectorizer.transform(texts)
            reduced = self._svd.transform(x)
            return l2_normalize(reduced.astype(np.float32))
        raise ValueError(self.backend)

    def encode_query(self, query: str) -> np.ndarray:
        return self.transform([query])[0]

    # -- backends -----------------------------------------------------------
    def _fit_minilm(self, texts: List[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        log.info("Loading sentence-transformers model '%s'.", self.cfg["embedding"]["model_name"])
        self._model = SentenceTransformer(self.cfg["embedding"]["model_name"])
        vecs = self._model.encode(
            texts, batch_size=self.cfg["embedding"]["batch_size"],
            show_progress_bar=True, normalize_embeddings=False,
        )
        vecs = l2_normalize(np.asarray(vecs, dtype=np.float32))
        self.dim = vecs.shape[1]
        return vecs

    def _fit_tfidf(self, texts: List[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        ecfg = self.cfg["embedding"]
        n_docs = len(texts)
        dims = min(int(ecfg["tfidf_dims"]), max(2, n_docs - 1))
        log.info("Fitting TF-IDF (max_features=%s) + SVD (dims=%d) on %d docs.",
                 ecfg["tfidf_max_features"], dims, n_docs)
        self._vectorizer = TfidfVectorizer(
            max_features=int(ecfg["tfidf_max_features"]),
            stop_words="english", ngram_range=(1, 2),
            min_df=2 if n_docs > 50 else 1, max_df=0.9,
        )
        x = self._vectorizer.fit_transform(texts)
        self._svd = TruncatedSVD(n_components=dims, random_state=self.cfg["seed"])
        reduced = self._svd.fit_transform(x)
        reduced = l2_normalize(reduced.astype(np.float32))
        self.dim = reduced.shape[1]
        return reduced

    # -- persistence --------------------------------------------------------
    def save(self, directory: str) -> None:
        ensure_dir(directory)
        meta = {"backend": self.backend, "dim": self.dim}
        with open(os.path.join(directory, "embedder_meta.pkl"), "wb") as fh:
            pickle.dump(meta, fh)
        if self.backend == "tfidf":
            with open(os.path.join(directory, "tfidf.pkl"), "wb") as fh:
                pickle.dump({"vectorizer": self._vectorizer, "svd": self._svd}, fh)
        # minilm: nothing to persist; reloaded by name from the model hub/cache.

    @classmethod
    def load(cls, directory: str, cfg: Config) -> "Embedder":
        with open(os.path.join(directory, "embedder_meta.pkl"), "rb") as fh:
            meta = pickle.load(fh)
        emb = cls(meta["backend"], cfg)
        emb.dim = meta["dim"]
        if emb.backend == "tfidf":
            with open(os.path.join(directory, "tfidf.pkl"), "rb") as fh:
                state = pickle.load(fh)
            emb._vectorizer = state["vectorizer"]
            emb._svd = state["svd"]
        elif emb.backend == "minilm":
            from sentence_transformers import SentenceTransformer

            emb._model = SentenceTransformer(cfg["embedding"]["model_name"])
        return emb
