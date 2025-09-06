from __future__ import annotations
from typing import List
import numpy as np

from ..models.schemas import PersonaCard, PersonaPreferences


def build_persona(
    user_id: str,
    sample_texts: List[str],
    exemplar_ids: List[str],
    preferences: PersonaPreferences,
    embedder,
) -> PersonaCard:
    """Compute a simple centroid embedding for the user's style and return a PersonaCard."""
    if sample_texts:
        embs = embedder.embed(sample_texts)
        centroid = embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroid_list = centroid.astype(float).tolist()
    else:
        centroid_list = None

    return PersonaCard(
        user_id=user_id,
        preferences=preferences,
        exemplar_ids=exemplar_ids,
        centroid=centroid_list,
    )

