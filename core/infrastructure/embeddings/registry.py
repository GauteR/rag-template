from __future__ import annotations

from collections.abc import Callable

from core.application.ports.embeddings import EmbedderPort
from core.config.settings import Settings
from core.infrastructure.embeddings.providers.hash import HashEmbedder
from core.infrastructure.embeddings.providers.ollama import OllamaEmbedder
from core.infrastructure.embeddings.providers.openai_compatible import (
    OpenAiCompatibleEmbedder,
)

EmbeddingFactory = Callable[[Settings], EmbedderPort]


class EmbeddingRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, EmbeddingFactory] = {}

    def register(self, provider_id: str, factory: EmbeddingFactory) -> None:
        self._factories[provider_id] = factory

    def provider_ids(self) -> set[str]:
        return set(self._factories)

    def build(self, provider_id: str, settings: Settings) -> EmbedderPort:
        try:
            return self._factories[provider_id](settings)
        except KeyError as exc:
            raise ValueError(f"Unknown embedding provider: {provider_id}") from exc


embedding_registry = EmbeddingRegistry()
embedding_registry.register(
    "hash",
    lambda settings: HashEmbedder(dimension=settings.embedding_dimension),
)
embedding_registry.register(
    "ollama",
    lambda settings: OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    ),
)
embedding_registry.register(
    "openai_compatible",
    lambda settings: OpenAiCompatibleEmbedder(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    ),
)
