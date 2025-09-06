from __future__ import annotations
import hashlib
import math
from typing import List

import numpy as np


class Embedder:
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """
    Lightweight, dependency-free hashing embedder. Not semantically strong,
    but good enough to power MVP retrieval and cosine scoring.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    def _hash_token(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
        return int(h, 16)

    def embed(self, texts: List[str]) -> np.ndarray:
        embs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok in self._tokenize(text):
                idx = self._hash_token(tok) % self.dim
                vec[idx] += 1.0
            # l2 normalize
            norm = math.sqrt(float((vec * vec).sum()))
            if norm > 0:
                vec /= norm
            embs[i] = vec
        return embs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class STEmbedder(Embedder):
    """
    Sentence-Transformers embedder (optional dependency).
    Enable via `uv sync --extra hf-embeddings` and set STYLGEN_EMBEDDER=st.
    """

    def __init__(self, model_name: str = "intfloat/e5-large-v2", normalize: bool = True):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError("sentence-transformers not installed; enable hf-embeddings extra") from e
        self._model = SentenceTransformer(model_name)
        self._normalize = normalize

    def embed(self, texts: List[str]) -> np.ndarray:
        # some models expect 'passage: ' or 'query: ' prefixes; we keep raw for style
        embs = self._model.encode(texts, normalize_embeddings=self._normalize, convert_to_numpy=True)
        return embs.astype(np.float32)
