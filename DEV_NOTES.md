# Developer Notes


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

