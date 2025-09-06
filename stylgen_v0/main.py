from __future__ import annotations
import uuid
from typing import List

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models.schemas import (
    HealthResponse,
    PersonaCreateRequest,
    PersonaCreateResponse,
    GenerationRequest,
    GenerationResponse,
    GenerationVariant,
    PersonaCard,
    FeedbackRequest,
)
from .core.embeddings import HashingEmbedder, STEmbedder
from .core.vector_store import InMemoryVectorStore, VecItem
from .core.persona import build_persona
from .core.pipeline import Pipeline
from .core.llm import OllamaProvider, DummyProvider
from .storage.memory import MemoryStore, WritingSample, GenerationRecord


app = FastAPI(title="stylgen_v0", version="0.1.0")

# Configure logging for this app
_log_level_name = os.getenv("STYLGEN_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)
logger = logging.getLogger("stylgen")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(_log_level)
logger.propagate = False

# Optional dev CORS (enable by setting STYLGEN_DEV_CORS=1)
if os.getenv("STYLGEN_DEV_CORS") == "1":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Single-process MVP instances
# Choose embedder: sentence-transformers if STYLGEN_EMBEDDER=st and available; else hashing
_embedder_choice = os.getenv("STYLGEN_EMBEDDER", "hash").lower()
if _embedder_choice in {"st", "hf", "sentence"}:
    _st_model = os.getenv("STYLGEN_ST_MODEL", "intfloat/e5-large-v2")
    try:
        embedder = STEmbedder(model_name=_st_model)
        logger.info("embedder.selected kind=st model=%s", _st_model)
    except Exception as e:
        logger.warning("embedder.st_unavailable falling back to hashing: %s", e)
        embedder = HashingEmbedder(dim=384)
        logger.info("embedder.selected kind=hash dim=%d", 384)
else:
    embedder = HashingEmbedder(dim=384)
    logger.info("embedder.selected kind=hash dim=%d", 384)
vstore = InMemoryVectorStore()
store = MemoryStore()

# Try Ollama, fallback to Dummy
try:
    llm = OllamaProvider()
except Exception:
    llm = DummyProvider()

pipeline = Pipeline(embedder=embedder, vstore=vstore, llm=llm)


@app.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health():
    return HealthResponse(status="ok")


@app.post("/persona", response_model=PersonaCreateResponse)
async def create_persona(req: PersonaCreateRequest):
    logger.info(
        "persona.create user_id=%s samples=%d",
        req.user_id,
        len(req.samples),
    )
    # store samples and build embeddings
    sample_ids: List[str] = []
    vec_items: List[VecItem] = []
    for text in req.samples:
        sid = str(uuid.uuid4())
        sample_ids.append(sid)
        vec = embedder.embed([text])[0]
        vec_items.append(VecItem(text=text, vec=vec, id=sid))

    # Replace existing vectors/samples for this user to avoid duplicates
    vstore.replace(req.user_id, vec_items)
    store.add_samples(req.user_id, [WritingSample(id=sid, text=txt) for sid, txt in zip(sample_ids, req.samples)])

    card: PersonaCard = build_persona(
        user_id=req.user_id,
        sample_texts=req.samples,
        exemplar_ids=sample_ids[:3],
        preferences=req.preferences,
        embedder=embedder,
    )

    store.set_persona(req.user_id, card.model_dump())
    logger.info(
        "persona.created user_id=%s exemplars=%d centroid=%s",
        req.user_id,
        len(card.exemplar_ids),
        "yes" if card.centroid is not None else "no",
    )

    return PersonaCreateResponse(user_id=req.user_id, num_samples=len(req.samples), persona=card)


@app.post("/generate", response_model=GenerationResponse)
async def generate(req: GenerationRequest):
    logger.info(
        "generate.request user_id=%s goal=%s keywords=%s variants=%d len_hint=%s",
        req.user_id,
        req.brief.goal,
        ",".join(req.brief.keywords),
        req.num_variants,
        req.brief.length_hint,
    )
    # get persona
    p = store.get_persona(req.user_id)
    if not p:
        raise HTTPException(status_code=400, detail="Persona not found for user. Create via /persona first.")

    persona = PersonaCard.model_validate(p)
    variants: List[GenerationVariant] = await pipeline.generate(
        persona=persona, brief=req.brief, k_exemplars=3, num_variants=req.num_variants, llm_options=req.llm_options or None
    )
    chosen = variants[0]

    gen_id = str(uuid.uuid4())
    rec = GenerationRecord(
        id=gen_id,
        user_id=req.user_id,
        variants=[v.model_dump() for v in variants],
        chosen_index=0,
    )
    store.add_generation(rec)
    logger.info(
        "generate.done user_id=%s generation_id=%s chosen_sim=%.3f chosen_novelty=%.3f",
        req.user_id,
        gen_id,
        variants[0].score.style_similarity,
        variants[0].score.novelty,
    )
    return GenerationResponse(user_id=req.user_id, generation_id=gen_id, chosen=chosen, variants=variants)


@app.post("/feedback")
async def feedback(data: FeedbackRequest):
    # Minimal placeholder to acknowledge feedback
    rec = store.get_generation(data.generation_id)
    if not rec:
        raise HTTPException(status_code=404, detail="generation_id not found")
    # Optionally: verify user ownership matches
    if rec.user_id != data.user_id:
        raise HTTPException(status_code=403, detail="generation_id does not belong to user_id")
    return {"status": "received"}


@app.post("/generate/stream")
async def generate_stream(req: GenerationRequest):
    logger.info(
        "generate.stream.request user_id=%s goal=%s keywords=%s",
        req.user_id,
        req.brief.goal,
        ",".join(req.brief.keywords),
    )
    p = store.get_persona(req.user_id)
    if not p:
        raise HTTPException(status_code=400, detail="Persona not found for user. Create via /persona first.")
    persona = PersonaCard.model_validate(p)

    # Prepare context (exemplars + system prompt)
    ex_texts, system = pipeline.prepare_generation_context(persona=persona, brief=req.brief, k_exemplars=3)
    prompt = pipeline._build_prompt(req.brief, ex_texts)

    # Select temperature
    llm_opts = req.llm_options or {}
    temp = float(llm_opts.get("temperature")) if "temperature" in llm_opts else 0.7

    async def event_source():
        # Send a small meta event first
        meta = {
            "exemplars": [t[:120] + ("…" if len(t) > 120 else "") for t in ex_texts],
            "goal": req.brief.goal,
            "keywords": req.brief.keywords,
        }
        yield f"event: meta\ndata: {meta}\n\n"
        try:
            async for chunk in pipeline.llm.stream_generate(prompt=prompt, system=system, temperature=temp, options=llm_opts):
                # Stream raw token chunks (no critique while streaming)
                safe = chunk.replace("\r", " ")
                yield f"data: {safe}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        finally:
            yield "event: done\ndata: end\n\n"

    from starlette.responses import StreamingResponse

    return StreamingResponse(event_source(), media_type="text/event-stream")
