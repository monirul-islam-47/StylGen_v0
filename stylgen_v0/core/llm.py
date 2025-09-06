from __future__ import annotations
import os
import json
import logging
from typing import Optional, AsyncGenerator, Dict, Any

import httpx


class LLMProvider:
    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7, options: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield chunks of generated text."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: float = 60.0):
        self.base_url = base_url or os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3:8b")
        self.timeout = timeout
        self.logger = logging.getLogger("stylgen.llm")

    def _merge_options(self, temperature: float, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = {"temperature": temperature}
        if options:
            merged.update(options)
        return merged

    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7, options: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt if system is None else f"<|system|>\n{system}\n<|user|>\n{prompt}",
            "stream": False,
            "options": self._merge_options(temperature, options),
        }
        url = f"{self.base_url}/api/generate"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            # Log tokens/sec if available
            try:
                eval_count = data.get("eval_count")
                eval_dur_ns = data.get("eval_duration")
                if eval_count and eval_dur_ns:
                    tps = float(eval_count) / (float(eval_dur_ns) / 1e9)
                    self.logger.debug("ollama.generate tokens=%s eval_ms=%.1f tps=%.1f", eval_count, float(eval_dur_ns) / 1e6, tps)
            except Exception:
                pass
            return data.get("response", "")

    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "prompt": prompt if system is None else f"<|system|>\n{system}\n<|user|>\n{prompt}",
            "stream": True,
            "options": self._merge_options(temperature, options),
        }
        url = f"{self.base_url}/api/generate"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # stream incremental text
                    chunk = data.get("response")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        # Log tokens/sec on completion
                        try:
                            eval_count = data.get("eval_count")
                            eval_dur_ns = data.get("eval_duration")
                            if eval_count and eval_dur_ns:
                                tps = float(eval_count) / (float(eval_dur_ns) / 1e9)
                                self.logger.debug(
                                    "ollama.stream tokens=%s eval_ms=%.1f tps=%.1f",
                                    eval_count,
                                    float(eval_dur_ns) / 1e6,
                                    tps,
                                )
                        except Exception:
                            pass
                        break


class DummyProvider(LLMProvider):
    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7, options: Optional[Dict[str, Any]] = None) -> str:
        # Fallback deterministic draft when no LLM is available
        return (
            "Hook: A concrete, personal detail that sets context.\n\n"
            "Body: Expand with one lesson and one example tied to the keywords."
            " Keep sentences short. Avoid clichés.\n\n"
            "CTA: Invite a specific action or comment.\n\n"
            "#niche1 #niche2 #niche3"
        )

    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        text = await self.generate(prompt=prompt, system=system, temperature=temperature, options=options)
        # Yield in a few chunks to simulate streaming
        parts = text.split("\n\n")
        for i, p in enumerate(parts):
            if i:
                yield "\n\n"
            yield p
