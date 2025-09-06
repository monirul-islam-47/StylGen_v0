from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class WritingSample:
    id: str
    text: str


@dataclass
class GenerationRecord:
    id: str
    user_id: str
    variants: List[dict]
    chosen_index: int


class MemoryStore:
    def __init__(self):
        # user -> samples
        self.samples: Dict[str, List[WritingSample]] = {}
        # user -> persona (dict)
        self.persona: Dict[str, dict] = {}
        # generation_id -> GenerationRecord
        self.generations: Dict[str, GenerationRecord] = {}

    def add_samples(self, user: str, items: List[WritingSample]) -> None:
        self.samples[user] = items

    def get_samples(self, user: str) -> List[WritingSample]:
        return list(self.samples.get(user, []))

    def set_persona(self, user: str, persona: dict) -> None:
        self.persona[user] = persona

    def get_persona(self, user: str) -> Optional[dict]:
        return self.persona.get(user)

    def add_generation(self, rec: GenerationRecord) -> None:
        self.generations[rec.id] = rec

    def get_generation(self, gen_id: str) -> Optional[GenerationRecord]:
        return self.generations.get(gen_id)

