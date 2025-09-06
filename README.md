# stylgen_v0

Minimal FastAPI backend (uv project) for personalized LinkedIn post generation using per-user Persona Cards, exemplar retrieval, and a simple two-step generation + critique pipeline.

For a deeper, team-focused guide covering architecture, LLM details, embeddings, streaming, and operations, see NOTES.md.

This is a skeleton you can run locally and extend. It supports:
- Creating/updating a Persona Card per user from samples + preferences
- Storing writing samples and building a simple vector index per user
- Generating a draft using an LLM provider (Ollama by default, with a lightweight fallback)
- Basic critique (ban phrases, enforce length) and simple scoring

## Quickstart

Prereqs:
- Python 3.10+
- uv (https://docs.astral.sh/uv/)
- Optional: Ollama running locally for best results (https://ollama.com)

Install deps with uv (from repo root):

```bash
uv sync
```

Run the API (dev):

```bash
uv run uvicorn stylgen_v0.main:app --reload --port 8000
```

Environment (optional):
- `OLLAMA_BASE` (default `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default `llama3:8b`)

## Endpoints

- `GET /health` → liveness probe
- `POST /persona` → Create/replace a user Persona Card and store samples
- `POST /generate` → Generate a LinkedIn-style post with persona conditioning
- `POST /generate/stream` → Stream tokens (SSE) for a live draft (raw, un-critiqued)
- `POST /feedback` → Submit ratings/tags/edits for a generation

See `examples/curl-examples.sh` for sample requests.

Example:

```bash
chmod +x examples/curl-examples.sh
./examples/curl-examples.sh http://127.0.0.1:8000
```

## Architecture Overview

- `stylgen_v0/main.py` – FastAPI app and routes
- `stylgen_v0/models/schemas.py` – Pydantic request/response models
- `stylgen_v0/core/persona.py` – Persona Card models + builder
- `stylgen_v0/core/embeddings.py` – Simple hashing embedder (no heavy deps) + interface
  - Optional: `STEmbedder` (sentence-transformers) when `STYLGEN_EMBEDDER=st`
- `stylgen_v0/core/vector_store.py` – In-memory vector store per user
- `stylgen_v0/core/llm.py` – LLM provider interface, Ollama provider, dummy fallback
- `stylgen_v0/core/pipeline.py` – Orchestrates retrieval → plan → draft → critique → score
- `stylgen_v0/storage/memory.py` – In-memory persistence for MVP

You can later plug in:
- vLLM / OpenAI-compatible endpoints as LLM provider
- Sentence-Transformers or other embedding backends (enable optional deps)
- Qdrant for vector storage
- Postgres/Redis for persistence/queues

## Notes
- The default embedder is a lightweight hashing TF-style vectorizer (cosine similarity). Replace with a real model for production.
- The default LLM provider is Ollama. If unavailable, the dummy provider creates a structured draft so the pipeline still works for demos.
- For dev-only CORS, set `STYLGEN_DEV_CORS=1` to allow all origins.
- Logging: control verbosity with `STYLGEN_LOG_LEVEL` (e.g., `DEBUG`, `INFO`). Set `STYLGEN_DEBUG=1` to log prompt/system previews and exemplar snippets.
- LLM options: pass per-request options in `/generate` as `llm_options` (e.g., `{ "temperature": 0.7, "top_p": 0.9, "num_predict": 512 }`).
- Streaming: `/generate/stream` returns Server-Sent Events (`text/event-stream`) with a `meta` event, token chunks as `data:`, and a final `done` event.
- Embeddings: to enable sentence-transformers, run `uv sync --extra hf-embeddings` and set `STYLGEN_EMBEDDER=st`. Default remains hashing.

To use Ollama:

```bash
# install Ollama if not installed, then:
export OLLAMA_BASE=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3:8b
# pull the model once
curl -X POST "$OLLAMA_BASE/api/pull" -d '{"name": "llama3:8b"}'
```

## Dev Tips
- Project uses uv. Add packages with `uv add <pkg>`; lock updated automatically.
- Run tests: `uv run pytest -q`
- Makefile helpers are available: `make help`
- The example script uses `jq` for pretty-printing; install it or remove the `| jq .` pipe.
- One-shot E2E: `make e2e-local` (starts server, waits for health, runs E2E, stops server)

## Responses
- `/generate` now returns a `generation_id` along with `chosen` and `variants`, which you can pass to `/feedback`.

## License
Proprietary / internal. Do not distribute.
