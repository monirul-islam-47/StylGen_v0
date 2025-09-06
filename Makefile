## stylgen_v0 Makefile (repo root)

.PHONY: help sync run run-prod pull-ollama examples e2e e2e-local

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
	@echo "  e2e          - Run end-to-end script (BASE=http://127.0.0.1:8000)"
	@echo "  e2e-local    - Start server, wait for health, run E2E, stop server"

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

BASE ?=
e2e:
	uv run python scripts/e2e.py $(BASE)

e2e-local:
	bash scripts/e2e_local.sh
