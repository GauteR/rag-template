from __future__ import annotations

import httpx

from core.infrastructure.llm.providers.openai_compatible import OpenAiCompatibleLlm


class OllamaLlm(OpenAiCompatibleLlm):
    def __init__(self, *, base_url: str, model: str) -> None:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            openai_base_url = normalized
        else:
            openai_base_url = f"{normalized}/v1"
        super().__init__(
            base_url=openai_base_url,
            api_key=None,
            model=model,
        )

    def _chat(self, content: str, *, json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        with httpx.Client(timeout=120) as client:
            response = client.post(f"{self._base_url}/chat/completions", json=payload)
            if response.status_code == 404:
                self._raise_model_not_found(response)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
