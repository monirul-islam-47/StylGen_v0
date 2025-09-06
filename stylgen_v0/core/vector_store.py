from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

from .embeddings import cosine


@dataclass
class VecItem:
    text: str
    vec: np.ndarray
    id: str


class InMemoryVectorStore:
    def __init__(self):
        # user_id -> List[VecItem]
        self._store: Dict[str, List[VecItem]] = {}

    def add(self, user_id: str, items: List[VecItem]) -> None:
        self._store.setdefault(user_id, []).extend(items)

    def clear(self, user_id: str) -> None:
        """Remove all vector items for a user."""
        if user_id in self._store:
            self._store[user_id] = []

    def replace(self, user_id: str, items: List[VecItem]) -> None:
        """Replace all vector items for a user in one call."""
        self._store[user_id] = list(items)

    def top_k(self, user_id: str, query_vec: np.ndarray, k: int = 3) -> List[Tuple[VecItem, float]]:
        items = self._store.get(user_id, [])
        scored = [(it, cosine(query_vec, it.vec)) for it in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def all(self, user_id: str) -> List[VecItem]:
        return list(self._store.get(user_id, []))
