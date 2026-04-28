from __future__ import annotations

import httpx

from core.application.ports.embeddings import EmbedderPort


class OllamaEmbedder(EmbedderPort):
    def __init__(self, *, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        with httpx.Client(timeout=30) as client:
            for text in texts:
                response = client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": text},
                )
                if response.status_code == 404:
                    response = client.post(
                        f"{self._base_url}/api/embeddings",
                        json={"model": self._model, "prompt": text},
                    )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload.get("embedding"), list):
                    embeddings.append(payload["embedding"])
                    continue
                if (
                    isinstance(payload.get("embeddings"), list)
                    and payload["embeddings"]
                    and isinstance(payload["embeddings"][0], list)
                ):
                    embeddings.append(payload["embeddings"][0])
                    continue
                raise ValueError("Unexpected Ollama embedding response format")
        return embeddings
