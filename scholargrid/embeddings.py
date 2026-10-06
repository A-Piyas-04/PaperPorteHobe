"""FR-02  Paper embedding generation.

A single :class:`Embedder` encodes both stored papers and user queries with the
*same* method (a hard FR-02 requirement). Two backends:

* ``minilm`` - a sentence-transformers model (default ``all-MiniLM-L6-v2``),
  optionally pinned to a Hugging Face revision so query embeddings in the app
  always match the corpus.
* ``tfidf``  - TfidfVectorizer -> TruncatedSVD -> L2 normalise. A fully
  reproducible, offline fallback that needs only scikit-learn.

For ``minilm`` vectors are cached per ``(arxiv_id, text hash, model, revision)``
so refreshes only embed new or changed papers. State is persisted as JSON
metadata plus a joblib file (tfidf only); bundles must come from a trusted
source because joblib files can execute code when loaded.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .utils import ensure_dir, get_logger, l2_normalize, load_json, save_json

log = get_logger("embed")


def paper_text(df: pd.DataFrame) -> List[str]:
    """FR-02: combine title + abstract."""
    return (df["title"].astype(str) + ". " + df["abstract"].astype(str)).tolist()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def resolve_device(want: str) -> str:
    if want != "auto":
        return want
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch missing
        return "cpu"


class Embedder:
    def __init__(self, backend: str, cfg: Config):
        self.backend = backend
        self.cfg = cfg
        ecfg = cfg["embedding"]
        self.model_name: str = ecfg["model_name"]
        self.revision: Optional[str] = ecfg.get("model_revision")
        self._model = None          # minilm
        self._vectorizer = None     # tfidf
        self._svd = None            # tfidf
        self.dim: int | None = None

    @property
    def fingerprint(self) -> str:
        if self.backend == "minilm":
            return f"{self.model_name}@{self.revision or 'latest'}"
        return "tfidf"

    # -- fit / encode -------------------------------------------------------
    def fit_transform(self, texts: List[str], ids: Optional[List[str]] = None) -> np.ndarray:
        if self.backend == "minilm":
            return self._fit_minilm(texts, ids)
        if self.backend == "tfidf":
            return self._fit_tfidf(texts)
        raise ValueError(f"Unknown embedding backend: {self.backend}")

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.backend == "minilm":
            self._ensure_model()
            vecs = self._model.encode(  # type: ignore[union-attr]
                texts, batch_size=self.cfg["embedding"]["batch_size"],
                show_progress_bar=False, normalize_embeddings=False,
            )
            return l2_normalize(np.asarray(vecs, dtype=np.float32))
        if self.backend == "tfidf":
            x = self._vectorizer.transform(texts)  # type: ignore[union-attr]
            return l2_normalize(self._svd.transform(x).astype(np.float32))  # type: ignore[union-attr]
        raise ValueError(self.backend)

    def encode_query(self, query: str) -> np.ndarray:
        return self.transform([query])[0]

    # -- backends -----------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        device = resolve_device(self.cfg["embedding"]["device"])
        log.info("Loading sentence-transformers model '%s' (revision=%s, device=%s).",
                 self.model_name, self.revision or "latest", device)
        self._model = SentenceTransformer(self.model_name, revision=self.revision, device=device)

    def _fit_minilm(self, texts: List[str], ids: Optional[List[str]]) -> np.ndarray:
        self._ensure_model()
        if ids is not None and self.cfg["embedding"]["incremental"]:
            vecs = _EmbeddingCache(self.cfg, self.fingerprint).encode(ids, texts, self.transform)
        else:
            vecs = self.transform(texts)
        self.dim = int(vecs.shape[1])
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
        reduced = l2_normalize(self._svd.fit_transform(x).astype(np.float32))
        self.dim = reduced.shape[1]
        return reduced

    # -- persistence --------------------------------------------------------
    def save(self, directory: str) -> None:
        ensure_dir(directory)
        save_json({"backend": self.backend, "dim": self.dim, "model_name": self.model_name,
                   "model_revision": self.revision}, os.path.join(directory, "embedder_meta.json"))
        if self.backend == "tfidf":
            import joblib

            joblib.dump({"vectorizer": self._vectorizer, "svd": self._svd},
                        os.path.join(directory, "tfidf.joblib"), compress=3)

    @classmethod
    def load(cls, directory: str, cfg: Config, lazy: bool = False) -> "Embedder":
        meta_json = os.path.join(directory, "embedder_meta.json")
        if os.path.exists(meta_json):
            meta = load_json(meta_json)
        else:  # bundles written before v2
            import pickle

            with open(os.path.join(directory, "embedder_meta.pkl"), "rb") as fh:
                meta = pickle.load(fh)
        emb = cls(meta["backend"], cfg)
        emb.dim = meta.get("dim")
        emb.model_name = meta.get("model_name") or emb.model_name
        emb.revision = meta.get("model_revision", emb.revision)
        if emb.backend == "tfidf":
            path = os.path.join(directory, "tfidf.joblib")
            if os.path.exists(path):
                import joblib

                state = joblib.load(path)
            else:
                import pickle

                with open(os.path.join(directory, "tfidf.pkl"), "rb") as fh:
                    state = pickle.load(fh)
            emb._vectorizer = state["vectorizer"]
            emb._svd = state["svd"]
        elif emb.backend == "minilm" and not lazy:
            emb._ensure_model()
        return emb


class _EmbeddingCache:
    """Vectors keyed by (arxiv_id, text hash) for one model fingerprint."""

    def __init__(self, cfg: Config, fingerprint: str):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", fingerprint)
        self.dir = os.path.join(cfg.stages_dir, "embedding_cache", safe)
        self.keys_path = os.path.join(self.dir, "keys.parquet")
        self.vec_path = os.path.join(self.dir, "vectors.npy")

    def encode(self, ids: List[str], texts: List[str], fn) -> np.ndarray:
        hashes = [text_hash(t) for t in texts]
        keys = [f"{i}:{h}" for i, h in zip(ids, hashes)]
        known: dict = {}
        vectors = None
        if os.path.exists(self.keys_path) and os.path.exists(self.vec_path):
            stored = pd.read_parquet(self.keys_path)["key"].tolist()
            vectors = np.load(self.vec_path)
            known = {k: j for j, k in enumerate(stored)}
        missing = [j for j, k in enumerate(keys) if k not in known]
        log.info("Embedding cache: %d reused, %d to embed.", len(keys) - len(missing), len(missing))
        new_vecs = fn([texts[j] for j in missing]) if missing else None
        dim = (new_vecs.shape[1] if new_vecs is not None else vectors.shape[1])  # type: ignore[union-attr]
        out = np.zeros((len(keys), dim), dtype=np.float32)
        for j, k in enumerate(keys):
            if k in known:
                out[j] = vectors[known[k]]  # type: ignore[index]
        if new_vecs is not None:
            out[missing] = new_vecs
            ensure_dir(self.dir)
            all_keys = list(known.keys()) + [keys[j] for j in missing]
            all_vecs = new_vecs if vectors is None else np.vstack([vectors, new_vecs])
            pd.DataFrame({"key": all_keys}).to_parquet(self.keys_path, index=False)
            np.save(self.vec_path, all_vecs.astype(np.float32))
        return out
