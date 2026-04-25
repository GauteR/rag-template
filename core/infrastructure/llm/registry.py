from __future__ import annotations

from collections.abc import Callable

from core.application.ports.llm import LlmPort
from core.config.settings import Settings
from core.infrastructure.llm.providers.anthropic import AnthropicLlm
from core.infrastructure.llm.providers.echo import EchoLlm
from core.infrastructure.llm.providers.ollama import OllamaLlm
from core.infrastructure.llm.providers.openai_compatible import OpenAiCompatibleLlm

LlmFactory = Callable[[Settings], LlmPort]


class LlmRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, LlmFactory] = {}

    def register(self, provider_id: str, factory: LlmFactory) -> None:
        self._factories[provider_id] = factory

    def provider_ids(self) -> set[str]:
        return set(self._factories)

    def build(self, provider_id: str, settings: Settings) -> LlmPort:
        try:
            return self._factories[provider_id](settings)
        except KeyError as exc:
            raise ValueError(f"Unknown LLM provider: {provider_id}") from exc


llm_registry = LlmRegistry()
llm_registry.register("echo", lambda _settings: EchoLlm())
llm_registry.register(
    "ollama",
    lambda settings: OllamaLlm(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
    ),
)
llm_registry.register(
    "openai_compatible",
    lambda settings: OpenAiCompatibleLlm(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
    ),
)
llm_registry.register(
    "anthropic",
    lambda settings: AnthropicLlm(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
    ),
)
