#!/usr/bin/env python3
import os
import sys
import json
import textwrap
import httpx


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else os.getenv("BASE_URL", "http://127.0.0.1:8000")
    user_id = os.getenv("E2E_USER", "u_e2e")

    print(f"Base URL: {base}")
    with httpx.Client(base_url=base, timeout=60.0) as client:
        # Health
        r = client.get("/health")
        r.raise_for_status()
        data = r.json()
        assert data.get("status") == "ok", f"Health not ok: {data}"
        print("✓ Health ok")

        # Persona
        persona_payload = {
            "user_id": user_id,
            "samples": [
                "Shipped our onboarding revamp. Short, friendly checklists beat long docs.",
                "If your standup drags, it's a smell. Keep it under 10 minutes, tops.",
                "Docs are a product. If you don't version them, they'll version you.",
            ],
            "preferences": {
                "tone_descriptors": ["forthright", "practical", "lightly humorous"],
                "taboo_phrases": ["In today's fast-paced world"],
                "formality": 2,
                "emoji_ok": True,
                "hashtags_niche": True,
                "structure_pref": "story-first",
            },
        }
        r = client.post("/persona", json=persona_payload)
        r.raise_for_status()
        pdata = r.json()
        assert pdata.get("user_id") == user_id and pdata.get("num_samples") == 3
        print("✓ Persona created/replaced")

        # Generate
        gen_payload = {
            "user_id": user_id,
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
        r = client.post("/generate", json=gen_payload)
        r.raise_for_status()
        g = r.json()
        gen_id = g.get("generation_id")
        assert isinstance(gen_id, str) and gen_id
        assert isinstance(g.get("variants"), list) and len(g["variants"]) == 2
        chosen = g["chosen"]["text"]
        print(f"✓ Generated 2 variants (generation_id={gen_id})")
        preview = textwrap.shorten(chosen.replace("\n", " / "), width=140)
        print(f"  Chosen preview: {preview}")
        print("  Variant scores:")
        for i, v in enumerate(g["variants"]):
            s = v["score"]
            print(
                f"    v{i}: style={s['style_similarity']:.3f} nov={s['novelty']:.3f} length_ok={s['length_ok']}"
            )

        # Feedback
        fb_payload = {
            "user_id": user_id,
            "generation_id": gen_id,
            "rating": 4,
            "tags": ["good tone"],
        }
        r = client.post("/feedback", json=fb_payload)
        r.raise_for_status()
        f = r.json()
        assert f.get("status") == "received"
        print("✓ Feedback accepted")

    print("E2E SUCCESS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                print("Response:", e.response.text)
            except Exception:
                pass
        raise SystemExit(1)
    except AssertionError as e:
        print("Assertion failed:", e)
        raise SystemExit(1)
    except Exception as e:
        print("Unexpected error:", repr(e))
        raise SystemExit(1)
