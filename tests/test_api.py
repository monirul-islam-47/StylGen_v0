import os
import sys
import pytest
import httpx

# Ensure project root is on sys.path when running via `uv run`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stylgen_v0.main import app, pipeline
from stylgen_v0.core.llm import DummyProvider


@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.mark.anyio
async def test_health_ok():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"


@pytest.mark.anyio
async def test_persona_generate_and_feedback_flow():
    # Force Dummy LLM to avoid external calls
    pipeline.llm = DummyProvider()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Create persona
        persona_payload = {
            "user_id": "u_test_1",
            "samples": [
                "Short, friendly checklists beat long docs.",
                "Keep standups under 10 minutes.",
                "Docs are a product; version them.",
            ],
            "preferences": {
                "tone_descriptors": ["forthright", "practical"],
                "taboo_phrases": ["In today's fast-paced world"],
                "formality": 2,
                "emoji_ok": True,
                "hashtags_niche": True,
                "structure_pref": "story-first",
            },
        }
        r = await client.post("/persona", json=persona_payload)
        assert r.status_code == 200
        pdata = r.json()
        assert pdata["user_id"] == "u_test_1"
        assert pdata["num_samples"] == 3
        assert "persona" in pdata

        # Generate
        gen_payload = {
            "user_id": "u_test_1",
            "brief": {
                "keywords": ["onboarding", "dev teams"],
                "goal": "educate",
                "audience": "engineering managers",
                "cta": "Comment with your experience",
                "length_hint": 900,
                "emoji": True,
            },
            "num_variants": 2,
        }
        r = await client.post("/generate", json=gen_payload)
        assert r.status_code == 200
        gdata = r.json()
        assert gdata["user_id"] == "u_test_1"
        assert isinstance(gdata["generation_id"], str) and gdata["generation_id"]
        assert "chosen" in gdata and "text" in gdata["chosen"] and "score" in gdata["chosen"]
        assert isinstance(gdata["variants"], list) and len(gdata["variants"]) == 2

        gen_id = gdata["generation_id"]

        # Feedback ok
        fb_payload = {
            "user_id": "u_test_1",
            "generation_id": gen_id,
            "rating": 4,
            "tags": ["good tone"],
        }
        r = await client.post("/feedback", json=fb_payload)
        assert r.status_code == 200
        assert r.json().get("status") == "received"

        # Feedback wrong user -> 403
        fb_wrong_user = {
            "user_id": "u_test_2",
            "generation_id": gen_id,
            "rating": 3,
        }
        r = await client.post("/feedback", json=fb_wrong_user)
        assert r.status_code == 403

        # Feedback unknown id -> 404
        fb_unknown = {
            "user_id": "u_test_1",
            "generation_id": "non-existent-id",
            "rating": 5,
        }
        r = await client.post("/feedback", json=fb_unknown)
        assert r.status_code == 404


@pytest.mark.anyio
async def test_stream_generate_dummy():
    # Force Dummy LLM to avoid external calls
    pipeline.llm = DummyProvider()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Ensure persona exists
        r = await client.post(
            "/persona",
            json={
                "user_id": "u_stream",
                "samples": ["A", "B"],
            },
        )
        assert r.status_code == 200

        # Stream request
        req = {
            "user_id": "u_stream",
            "brief": {"keywords": ["x"], "goal": "educate"},
        }
        # httpx streaming with ASGITransport via 'stream' context
        async with client.stream("POST", "/generate/stream", json=req) as resp:
            assert resp.status_code == 200
            # Read a small portion to verify SSE format
            chunks = []
            async for chunk in resp.aiter_text():
                chunks.append(chunk)
                if len("".join(chunks)) > 50:
                    break
            body = "".join(chunks)
            assert "data:" in body or "event: meta" in body
