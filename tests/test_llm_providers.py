import httpx
import pytest

from core.config.settings import Settings
from core.domain.models import Section
from core.infrastructure.llm.providers.anthropic import AnthropicLlm
from core.infrastructure.llm.providers.echo import EchoLlm
from core.infrastructure.llm.providers.ollama import OllamaLlm
from core.infrastructure.llm.providers.openai_compatible import OpenAiCompatibleLlm
from core.infrastructure.llm.registry import llm_registry


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(self._payload)
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


class _RecordingClient:
    def __init__(self, *, responses: list[_FakeResponse], timeout: int) -> None:
        self.timeout = timeout
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self) -> "_RecordingClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict, **kwargs) -> _FakeResponse:
        self.calls.append((url, json))
        if not self.responses:
            raise AssertionError(f"No fake response left for POST {url}")
        return self.responses.pop(0)


def _section(node_id: str = "doc:n1") -> Section:
    return Section(
        doc_id="doc",
        node_id=node_id,
        breadcrumb=("Intro",),
        text="Install with uv sync",
    )


def test_echo_llm_synthesize_and_rerank_without_http() -> None:
    llm = EchoLlm()
    sections = [_section(), _section("doc:n2")]

    assert "Install with uv sync" in llm.synthesize(question="How to install?", sections=sections)
    assert llm.rerank(question="install", candidates=sections, k_final=1) == ["doc:n1"]
    assert llm.filter_noise(sections=sections) == set()
    assert list(llm.synthesize_stream(question="q", sections=sections)) == [
        llm.synthesize(question="q", sections=sections)
    ]


def test_openai_compatible_uses_json_mode_for_rerank(monkeypatch) -> None:
    fake_client = _RecordingClient(
        timeout=60,
        responses=[
            _FakeResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": '{"node_ids": ["doc:n1"]}'}}]},
            )
        ],
    )
    monkeypatch.setattr(
        "core.infrastructure.llm.providers.openai_compatible.httpx.Client",
        lambda timeout, headers: fake_client,
    )

    llm = OpenAiCompatibleLlm(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-test",
    )
    ranked = llm.rerank(question="install?", candidates=[_section()], k_final=1)

    assert ranked == ["doc:n1"]
    assert fake_client.calls[0][0] == "https://api.example.com/v1/chat/completions"
    assert fake_client.calls[0][1]["response_format"] == {"type": "json_object"}
    assert fake_client.calls[0][1]["messages"][0]["content"].startswith("{")


def test_openai_compatible_raises_clear_error_for_missing_model(monkeypatch) -> None:
    fake_client = _RecordingClient(
        timeout=60,
        responses=[
            _FakeResponse(
                status_code=404,
                payload={"error": {"message": "model 'missing' not found"}},
            )
        ],
    )
    monkeypatch.setattr(
        "core.infrastructure.llm.providers.openai_compatible.httpx.Client",
        lambda timeout, headers: fake_client,
    )
    llm = OpenAiCompatibleLlm(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="missing",
    )

    with pytest.raises(ValueError, match="model 'missing' not found"):
        llm.synthesize(question="hello", sections=[_section()])


def test_ollama_llm_normalizes_base_url_and_uses_json_format(monkeypatch) -> None:
    fake_client = _RecordingClient(
        timeout=120,
        responses=[
            _FakeResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": '{"noisy_node_ids": []}'}}]},
            )
        ],
    )
    monkeypatch.setattr(
        "core.infrastructure.llm.providers.ollama.httpx.Client",
        lambda timeout: fake_client,
    )

    llm = OllamaLlm(base_url="http://localhost:11434/", model="llama3.1")
    assert llm._base_url == "http://localhost:11434/v1"
    assert llm.filter_noise(sections=[_section()]) == set()

    assert fake_client.calls[0][0] == "http://localhost:11434/v1/chat/completions"
    assert fake_client.calls[0][1]["format"] == "json"
    assert fake_client.calls[0][1]["stream"] is False


def test_ollama_llm_accepts_base_url_with_existing_v1_suffix(monkeypatch) -> None:
    fake_client = _RecordingClient(
        timeout=120,
        responses=[
            _FakeResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "answer"}}]},
            )
        ],
    )
    monkeypatch.setattr(
        "core.infrastructure.llm.providers.ollama.httpx.Client",
        lambda timeout: fake_client,
    )

    llm = OllamaLlm(base_url="http://localhost:11434/v1", model="llama3.1")
    assert llm._base_url == "http://localhost:11434/v1"
    assert llm.synthesize(question="hello", sections=[_section()]) == "answer"


def test_anthropic_llm_parses_rerank_json(monkeypatch) -> None:
    fake_client = _RecordingClient(
        timeout=60,
        responses=[
            _FakeResponse(
                status_code=200,
                payload={"content": [{"text": '{"node_ids": ["doc:n1"]}'}]},
            )
        ],
    )
    monkeypatch.setattr(
        "core.infrastructure.llm.providers.anthropic.httpx.Client",
        lambda timeout: fake_client,
    )

    llm = AnthropicLlm(api_key="secret", model="claude-test")
    ranked = llm.rerank(question="install?", candidates=[_section()], k_final=1)

    assert ranked == ["doc:n1"]
    assert fake_client.calls[0][0] == "https://api.anthropic.com/v1/messages"


def test_llm_registry_builds_all_providers() -> None:
    settings = Settings(
        llm_provider="echo",
        openai_api_key="test",
        anthropic_api_key="test",
    )
    for provider_id in llm_registry.provider_ids():
        llm = llm_registry.build(provider_id, settings)
        assert llm is not None
