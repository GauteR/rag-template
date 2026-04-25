from __future__ import annotations

import httpx

from rag_template_core.application.ports.embeddings import EmbedderPort


class OllamaEmbedder(EmbedderPort):
    def __init__(self, *, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        with httpx.Client(timeout=30) as client:
            for text in texts:
                response = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
        return embeddings
