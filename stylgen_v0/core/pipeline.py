from __future__ import annotations
from typing import List, Tuple
import os
import re
import time
import logging
import numpy as np

from .embeddings import HashingEmbedder, cosine
from .vector_store import InMemoryVectorStore, VecItem
from .llm import LLMProvider, OllamaProvider, DummyProvider
from ..models.schemas import (
    PersonaCard,
    GenerationBrief,
    GenerationVariant,
    VariantScore,
)


DEFAULT_BANNED = {
    "as an ai",
    "in today's fast-paced world",
    "game-changer",
    "cutting-edge",
    "unlock your potential",
}


class Pipeline:
    def __init__(
        self,
        embedder: HashingEmbedder,
        vstore: InMemoryVectorStore,
        llm: LLMProvider | None = None,
    ):
        self.embedder = embedder
        self.vstore = vstore
        self.llm = llm or self._default_llm()
        self.logger = logging.getLogger("stylgen.pipeline")
        self.debug = os.getenv("STYLGEN_DEBUG") == "1"

    def _default_llm(self) -> LLMProvider:
        # Prefer Ollama; fallback to Dummy
        try:
            return OllamaProvider()
        except Exception:
            return DummyProvider()

    def _build_system(self, persona: PersonaCard) -> str:
        pref = persona.preferences
        tone = ", ".join(pref.tone_descriptors) if pref.tone_descriptors else "authentic, specific"
        formality = f"Formality: {pref.formality or 3}/5"
        emoji = "Use emojis sparingly" if pref.emoji_ok else "Do not use emojis"
        hashtags = "Use 3–5 niche hashtags" if pref.hashtags_niche else "Use up to 3 common hashtags"
        taboo = set(DEFAULT_BANNED) | set(map(str.lower, pref.taboo_phrases or []))
        taboo_txt = "; ".join(sorted(taboo))
        structure = pref.structure_pref or "story-first"
        return (
            "You are a writing partner specialized in capturing a specific person’s voice for LinkedIn.\n"
            f"Tone: {tone}. {formality}. {emoji}. {hashtags}.\n"
            f"Structure preference: {structure}.\n"
            f"Avoid clichés and banned phrases: {taboo_txt}.\n"
            "Write in first-person, keep sentences concise, include one concrete detail and a clear CTA."
        )

    def _build_prompt(self, brief: GenerationBrief, exemplars: List[str]) -> str:
        kw = ", ".join(brief.keywords)
        aud = brief.audience or "LinkedIn audience"
        cta = brief.cta or "Comment with your experience."
        length = brief.length_hint or 1000
        use_emoji = brief.emoji if brief.emoji is not None else True
        ex_str = "\n\n".join([f"Example: {e}" for e in exemplars]) if exemplars else ""
        return (
            f"Goal: {brief.goal}. Audience: {aud}. Keywords: {kw}.\n"
            f"Constraints: length ~{length} characters; {'allow' if use_emoji else 'no'} emojis; 3–5 niche hashtags.\n"
            f"CTA: {cta}. {'Link: ' + brief.link if brief.link else ''}\n"
            f"{ex_str}\n\n"
            "Produce a single LinkedIn post draft that matches the tone and constraints."
        )

    def prepare_generation_context(self, persona: PersonaCard, brief: GenerationBrief, k_exemplars: int = 3) -> tuple[list[str], str]:
        """Return exemplar texts and system prompt for a request."""
        ex_texts: List[str] = []
        if persona.exemplar_ids:
            items = self.vstore.all(persona.user_id)
            id_map = {it.id: it for it in items}
            for eid in persona.exemplar_ids:
                if eid in id_map:
                    ex_texts.append(id_map[eid].text)
        if not ex_texts and persona.centroid is not None:
            query_vec = np.array(persona.centroid, dtype=np.float32)
            top = self.vstore.top_k(persona.user_id, query_vec, k=k_exemplars)
            ex_texts = [it.text for (it, _s) in top]
        system = self._build_system(persona)
        return ex_texts, system

    def _critique(self, text: str, persona: PersonaCard, brief: GenerationBrief) -> Tuple[str, dict]:
        """Apply simple rules: remove banned phrases, enforce soft length, normalize whitespace."""
        taboo = set(DEFAULT_BANNED) | set(map(str.lower, persona.preferences.taboo_phrases or []))
        fixed = text
        for phrase in taboo:
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)
            fixed = pattern.sub("", fixed)
        # normalize spaces/tabs but preserve line breaks
        fixed = re.sub(r"[ \t]+", " ", fixed).strip()
        # length check
        target = brief.length_hint or 1000
        length_ok = 0.6 * target <= len(fixed) <= 1.4 * target
        return fixed, {"length_ok": length_ok}

    def _novelty(self, user_embs: np.ndarray, cand_vec: np.ndarray) -> float:
        if user_embs.size == 0:
            return 1.0
        sims = user_embs @ cand_vec / (
            (np.linalg.norm(user_embs, axis=1) * (np.linalg.norm(cand_vec) + 1e-8)) + 1e-8
        )
        # novelty = 1 - max similarity to any existing sample
        return float(1.0 - max(0.0, float(sims.max())))

    async def generate(self, persona: PersonaCard, brief: GenerationBrief, k_exemplars: int = 3, num_variants: int = 2, llm_options: dict | None = None) -> List[GenerationVariant]:
        ex_texts, system = self.prepare_generation_context(persona, brief, k_exemplars=k_exemplars)
        if self.debug:
            self.logger.debug("system: %s", system[:300] + ("…" if len(system) > 300 else ""))
        if self.debug and ex_texts:
            previews = [t.replace("\n", " ")[:120] + ("…" if len(t) > 120 else "") for t in ex_texts]
            self.logger.debug("exemplars.count=%d previews=%s", len(ex_texts), previews)
        variants: List[GenerationVariant] = []

        # Precompute user embedding matrix for novelty if available
        user_items = self.vstore.all(persona.user_id)
        user_embs = np.stack([it.vec for it in user_items]) if user_items else np.zeros((0, self.embedder.dim), dtype=np.float32)

        for i in range(max(1, num_variants)):
            prompt = self._build_prompt(brief, ex_texts)
            if self.debug:
                self.logger.debug("prompt[%d]: %s", i, prompt[:300] + ("…" if len(prompt) > 300 else ""))
            # Slightly vary temperature per variant
            temp = (
                float(llm_options.get("temperature")) if llm_options and "temperature" in llm_options else 0.6 + 0.2 * (i % 2)
            )
            t0 = time.time()
            try:
                text = await self.llm.generate(prompt=prompt, system=system, temperature=temp, options=llm_options)
            except Exception:
                self.logger.warning("llm.generate failed; falling back to DummyProvider for variant=%d", i)
                text = await DummyProvider().generate(prompt=prompt, system=system)
            llm_ms = int((time.time() - t0) * 1000)

            fixed, checks = self._critique(text, persona, brief)
            cand_vec = self.embedder.embed([fixed])[0]
            # style similarity to centroid
            if persona.centroid is not None:
                sim = cosine(cand_vec, np.array(persona.centroid, dtype=np.float32))
            else:
                sim = 0.0
            novelty = self._novelty(user_embs, cand_vec)
            v = GenerationVariant(
                text=fixed,
                score=VariantScore(
                    style_similarity=float(sim),
                    novelty=float(novelty),
                    structure_ok=True,
                    length_ok=bool(checks.get("length_ok", True)),
                ),
            )
            variants.append(v)
            self.logger.debug(
                "variant[%d] temp=%.2f llm_ms=%d sim=%.3f nov=%.3f length_ok=%s",
                i,
                temp,
                llm_ms,
                v.score.style_similarity,
                v.score.novelty,
                v.score.length_ok,
            )

        # sort by composite score: prioritize style + novelty, then length_ok
        variants.sort(key=lambda x: (x.score.style_similarity + 0.5 * x.score.novelty, 1.0 if x.score.length_ok else 0.0), reverse=True)
        if variants:
            top = variants[0].score
            self.logger.info(
                "variants.sorted count=%d top_sim=%.3f top_nov=%.3f length_ok=%s",
                len(variants),
                top.style_similarity,
                top.novelty,
                top.length_ok,
            )
        return variants
