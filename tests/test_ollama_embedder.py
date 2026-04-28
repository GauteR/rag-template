import httpx

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
