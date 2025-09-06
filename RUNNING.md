# Running stylgen_v0 (uv project)

This guide shows how to set up and run the minimal FastAPI backend that powers persona-based LinkedIn post generation.

## Prerequisites
- Python 3.10+
- uv (package/dependency manager)
- Optional: Ollama running locally for best LLM results

Install uv (choose one):
- Using pipx: `pipx install uv`
- Using pip: `python -m pip install --user uv`
- Docs: https://docs.astral.sh/uv/

## Setup
```bash
uv sync
```

## Run the API (development)
```bash
uv run uvicorn stylgen_v0.main:app --reload --port 8000
```
- App docs (Swagger): http://127.0.0.1:8000/docs
- Health check: `GET /health`

## Configure Ollama (optional but recommended)
If you have Ollama installed locally:
```bash
# default env (adjust as needed)
export OLLAMA_BASE=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3:8b

# pull the model once (or pick another model you prefer)
curl -X POST "$OLLAMA_BASE/api/pull" -H 'content-type: application/json' \
     -d '{"name": "llama3:8b"}'
```
If Ollama isn’t available, the app uses a deterministic dummy provider so you can still exercise the pipeline.

## Try It With cURL
Use the provided script:
```bash
chmod +x examples/curl-examples.sh
./examples/curl-examples.sh http://127.0.0.1:8000
```

Or manual calls:
```bash
# 1) Create a persona
curl -s -X POST http://127.0.0.1:8000/persona \
  -H 'content-type: application/json' -d @- <<'JSON'
{
  "user_id": "u1",
  "samples": [
    "Shipped our onboarding revamp. Short, friendly checklists beat long docs.",
    "If your standup drags, it's a smell. Keep it under 10 minutes, tops.",
    "Docs are a product. If you don't version them, they'll version you."
  ],
  "preferences": {
    "tone_descriptors": ["forthright", "practical", "lightly humorous"],
    "taboo_phrases": ["In today's fast-paced world"],
    "formality": 2,
    "emoji_ok": true,
    "hashtags_niche": true,
    "structure_pref": "story-first"
  }
}
JSON

# 2) Generate a post
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'content-type: application/json' -d @- <<'JSON'
{
  "user_id": "u1",
  "brief": {
    "keywords": ["onboarding", "dev teams"],
    "goal": "educate",
    "audience": "engineering managers",
    "cta": "Comment with your experience",
    "length_hint": 900,
    "emoji": true
  },
  "num_variants": 2
}
JSON
```

## Endpoints
- `GET /health` → liveness probe
- `POST /persona` → Create/replace Persona Card and store samples
- `POST /generate` → Generate LinkedIn-style post(s) using persona + exemplars
- `POST /generate/stream` → Stream tokens (SSE) for a live draft (raw, un-critiqued)
- `POST /feedback` → Accept feedback for later learning (stub in MVP)

## Project Structure (key files)
- `stylgen_v0/main.py` – FastAPI routes and wiring
- `stylgen_v0/models/schemas.py` – request/response models
- `stylgen_v0/core/embeddings.py` – lightweight hashing embedder (cosine)
  - Optional: `STEmbedder` (sentence-transformers) when extra is installed
- `stylgen_v0/core/vector_store.py` – in-memory per-user vectors
- `stylgen_v0/core/persona.py` – Persona builder (centroid from samples)
- `stylgen_v0/core/llm.py` – Ollama + Dummy providers
- `stylgen_v0/core/pipeline.py` – retrieval → prompt → generate → critique → score
- `stylgen_v0/storage/memory.py` – simple in-memory store

## Notes & Next Steps
- Embeddings: swap `HashingEmbedder` for `sentence-transformers` (enable optional deps in `pyproject.toml`).
- Vector DB: replace `InMemoryVectorStore` with Qdrant for persistence and scale.
- LLM backends: add vLLM/OpenAI-compatible provider; keep interface in `core/llm.py`.
- Persistence: add Postgres/Redis for personas, samples, and generations.
- Critique: expand checklist (banned phrases per user, structure/readability checks).
- Dev CORS: set `STYLGEN_DEV_CORS=1` to allow all origins during local frontend testing.
- Per-request LLM options: send `llm_options` in `/generate` to tune temperature, top_p, num_predict, etc.
- Streaming: use `/generate/stream` for live token output (SSE).

## Troubleshooting
- `Connection refused` to Ollama: ensure `ollama serve` is running and `OLLAMA_BASE` points to it.
- `ModuleNotFoundError` for deps: re-run `uv sync` from the project root.
- CORS (browser): add `fastapi.middleware.cors` with allowed origins during frontend integration.
- `jq` not found: either install `jq` or remove `| jq .` from example commands.
- Sentence-Transformers not found: run `uv sync --extra hf-embeddings` and set `STYLGEN_EMBEDDER=st`.
