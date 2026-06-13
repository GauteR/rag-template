import httpx
import pytest

from core.infrastructure.embeddings.providers.ollama import OllamaEmbedder


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.request = httpx.Request("POST", "http://test")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, *, timeout: int) -> None:
        self.timeout = timeout
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict) -> _FakeResponse:
        self.calls.append((url, json))
        if url.endswith("/api/embed"):
            return _FakeResponse(status_code=404, payload={})
        if url.endswith("/api/embeddings"):
            return _FakeResponse(status_code=200, payload={"embedding": [0.1, 0.2, 0.3]})
        return _FakeResponse(status_code=500, payload={})


def test_ollama_embedder_falls_back_to_legacy_embeddings_endpoint(monkeypatch) -> None:
    fake_client = _FakeClient(timeout=30)
    monkeypatch.setattr(
        "core.infrastructure.embeddings.providers.ollama.httpx.Client",
        lambda timeout: fake_client,
    )

    embedder = OllamaEmbedder(base_url="http://localhost:11434", model="nomic-embed-text")

    embeddings = embedder.embed_texts(["hei"])

    assert embeddings == [[0.1, 0.2, 0.3]]
    assert [call[0] for call in fake_client.calls] == [
        "http://localhost:11434/api/embed",
        "http://localhost:11434/api/embeddings",
    ]


def test_ollama_embedder_parses_new_embeddings_array_format(monkeypatch) -> None:
    class _EmbedClient(_FakeClient):
        def post(self, url: str, json: dict) -> _FakeResponse:
            self.calls.append((url, json))
            return _FakeResponse(
                status_code=200,
                payload={"embeddings": [[0.4, 0.5, 0.6]]},
            )

    fake_client = _EmbedClient(timeout=30)
    monkeypatch.setattr(
        "core.infrastructure.embeddings.providers.ollama.httpx.Client",
        lambda timeout: fake_client,
    )

    embedder = OllamaEmbedder(base_url="http://localhost:11434", model="nomic-embed-text")
    assert embedder.embed_texts(["hei"]) == [[0.4, 0.5, 0.6]]


def test_ollama_embedder_does_not_fallback_when_model_is_missing(monkeypatch) -> None:
    class _MissingModelClient(_FakeClient):
        def post(self, url: str, json: dict) -> _FakeResponse:
            self.calls.append((url, json))
            return _FakeResponse(
                status_code=404,
                payload={"error": 'model "nomic-embed-text" not found, try pulling it first'},
            )

    fake_client = _MissingModelClient(timeout=30)
    monkeypatch.setattr(
        "core.infrastructure.embeddings.providers.ollama.httpx.Client",
        lambda timeout: fake_client,
    )

    embedder = OllamaEmbedder(base_url="http://localhost:11434", model="nomic-embed-text")

    with pytest.raises(ValueError, match='model "nomic-embed-text" not found'):
        embedder.embed_texts(["hei"])

    assert len(fake_client.calls) == 1
    assert fake_client.calls[0][0].endswith("/api/embed")
