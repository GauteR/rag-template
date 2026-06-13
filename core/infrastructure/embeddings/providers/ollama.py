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
                if response.status_code == 404 and self._should_fallback_to_legacy(response):
                    response = client.post(
                        f"{self._base_url}/api/embeddings",
                        json={"model": self._model, "prompt": text},
                    )
                if response.status_code == 404:
                    self._raise_model_not_found(response)
                response.raise_for_status()
                embeddings.append(self._parse_embedding_payload(response.json()))
        return embeddings

    def _should_fallback_to_legacy(self, response: httpx.Response) -> bool:
        try:
            payload = response.json()
        except ValueError:
            return True
        error = payload.get("error")
        if isinstance(error, str) and "try pulling it first" in error:
            return False
        return True

    def _parse_embedding_payload(self, payload: dict) -> list[float]:
        if isinstance(payload.get("embedding"), list):
            return payload["embedding"]
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        raise ValueError("Unexpected Ollama embedding response format")

    def _raise_model_not_found(self, response: httpx.Response) -> None:
        try:
            payload = response.json()
            error = payload.get("error")
            if isinstance(error, str):
                raise ValueError(f"Ollama embedding request failed: {error}") from None
        except ValueError:
            raise
        response.raise_for_status()
