#!/usr/bin/env bash
set -euo pipefail

base=${1:-http://127.0.0.1:8000}

echo 'Health:'
curl -s "$base/health" | jq .

echo 'Create persona:'
curl -s -X POST "$base/persona" -H 'content-type: application/json' -d @- <<'JSON' | jq .
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

echo 'Generate:'
curl -s -X POST "$base/generate" -H 'content-type: application/json' -d @- <<'JSON' | jq .
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
  "num_variants": 2,
  "llm_options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 512}
}
JSON

echo 'Stream (SSE):'
curl -N -s -X POST "$base/generate/stream" -H 'content-type: application/json' -d @- <<'JSON'
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
  "num_variants": 1,
  "llm_options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 256}
}
JSON
