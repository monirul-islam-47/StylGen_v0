from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class PersonaPreferences(BaseModel):
    tone_descriptors: List[str] = Field(default_factory=list)
    taboo_phrases: List[str] = Field(default_factory=list)
    formality: Optional[int] = Field(default=None, ge=1, le=5)
    emoji_ok: bool = True
    hashtags_niche: bool = True
    structure_pref: Optional[str] = None  # e.g., "story-first", "data-first", "contrarian"


class PersonaCard(BaseModel):
    user_id: str
    preferences: PersonaPreferences
    exemplar_ids: List[str] = Field(default_factory=list)
    centroid: Optional[List[float]] = None  # embedding centroid for style similarity


class PersonaCreateRequest(BaseModel):
    user_id: str
    samples: List[str]
    preferences: PersonaPreferences = Field(default_factory=PersonaPreferences)


class PersonaCreateResponse(BaseModel):
    user_id: str
    num_samples: int
    persona: PersonaCard


class GenerationBrief(BaseModel):
    keywords: List[str]
    goal: str  # e.g., "educate", "announce", "story"
    audience: Optional[str] = None
    cta: Optional[str] = None
    length_hint: Optional[int] = 1000  # characters
    emoji: Optional[bool] = None
    link: Optional[str] = None


class GenerationRequest(BaseModel):
    user_id: str
    brief: GenerationBrief
    num_variants: int = 2
    llm_options: Optional[Dict[str, float | int]] = None  # e.g., {"temperature": 0.7, "top_p": 0.9, "num_predict": 512}


class VariantScore(BaseModel):
    style_similarity: float
    novelty: float
    structure_ok: bool
    length_ok: bool


class GenerationVariant(BaseModel):
    text: str
    score: VariantScore


class GenerationResponse(BaseModel):
    user_id: str
    generation_id: str
    chosen: GenerationVariant
    variants: List[GenerationVariant]


class FeedbackRequest(BaseModel):
    user_id: str
    generation_id: str
    rating: int = Field(ge=1, le=5)
    tags: List[str] = Field(default_factory=list)
    edit_diff: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
