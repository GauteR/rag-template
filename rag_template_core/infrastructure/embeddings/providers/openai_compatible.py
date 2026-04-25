from __future__ import annotations

import httpx

from rag_template_core.application.ports.embeddings import EmbedderPort


class OpenAiCompatibleEmbedder(EmbedderPort):
    def __init__(self, *, base_url: str, api_key: str | None, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        with httpx.Client(timeout=30, headers=headers) as client:
            response = client.post(
                f"{self._base_url}/embeddings",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in data]
