"""Optional OpenAI-compatible HTTP callbacks; no agent or generation loop."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


def _post(base_url: str, path: str, api_key: str, payload: dict,
          timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"memory provider returned HTTP {exc.code} for {path}") from exc


@dataclass
class ChatClient:
    """JSON completion callback for extracting feedback, not generating slides.

    ``base_url`` is the provider API root, including ``/v1`` when required.
    Each call uses a fresh message list. Supply your own callback instead when
    the host already has a model client.
    """

    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0

    def __call__(self, prompt: str) -> dict:
        response = _post(self.base_url, "chat/completions", self.api_key, {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }, self.timeout)
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("semantic provider must return JSON text")
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        proposal = json.loads(content)
        if not isinstance(proposal, dict):
            raise ValueError("semantic provider must return a JSON object")
        return proposal


@dataclass
class EmbeddingClient:
    """Embedding callback. Use the same model for writing and retrieving."""

    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0

    def __call__(self, text: str) -> list[float]:
        response = _post(self.base_url, "embeddings", self.api_key, {
            "model": self.model, "input": text,
        }, self.timeout)
        return response["data"][0]["embedding"]
