# Developer Notes

## What I was doing
- Adding a `Makefile` to streamline common tasks for this uv-based FastAPI project (sync deps, run dev/prod server, pull an Ollama model, show example commands).
- The last attempt to create `stylgen_v0/Makefile` was interrupted. Keeping the exact content here so it can be added later with a single paste.

## Proposed Makefile content
```Makefile
# stylgen_v0 Makefile

.PHONY: help sync run run-prod pull-ollama examples

PORT ?= 8000
HOST ?= 127.0.0.1

# Ollama config (override as needed)
OLLAMA_BASE ?= http://127.0.0.1:11434
OLLAMA_MODEL ?= llama3:8b

help:
	@echo "Targets:"
	@echo "  sync         - Install/lock deps with uv"
	@echo "  run          - Run dev server with reload (PORT=$(PORT))"
	@echo "  run-prod     - Run server without reload (HOST=$(HOST) PORT=$(PORT))"
	@echo "  pull-ollama  - Pull OLLAMA_MODEL=$(OLLAMA_MODEL) from OLLAMA_BASE=$(OLLAMA_BASE)"
	@echo "  examples     - Show curl example command"

sync:
	uv sync

run:
	uv run uvicorn stylgen_v0.main:app --reload --port $(PORT)

run-prod:
	uv run uvicorn stylgen_v0.main:app --host $(HOST) --port $(PORT)

pull-ollama:
	curl -X POST "$(OLLAMA_BASE)/api/pull" -H 'content-type: application/json' -d '{"name": "$(OLLAMA_MODEL)"}'

examples:
	@echo "Run: chmod +x examples/curl-examples.sh && ./examples/curl-examples.sh http://127.0.0.1:$(PORT)"
```

## Next steps (nice to have)
- Embeddings: replace `HashingEmbedder` with `sentence-transformers` (enable `hf-embeddings` extra) and optionally add a Qdrant-backed vector store.
- LLM providers: add a vLLM/OpenAI-compatible provider alongside Ollama; allow selecting provider via env.
- Critique: expand checklist (banned phrases per user, structure/readability checks, CTA detection) and add configurable thresholds.
- Persistence: swap in Postgres for personas/samples/generations and Redis for queues/caching.
- Auth & CORS: add basic auth and CORS middleware for frontend integration.
- Tests: add unit tests for persona builder, scoring, and pipeline orchestration.

```
To add the Makefile later:
  1) Create stylgen_v0/Makefile with the exact content above.
  2) Use: make sync; make run; make pull-ollama OLLAMA_MODEL=llama3:8b
```

