from __future__ import annotations

from collections.abc import Callable

from core.application.ports.section_source import SectionSourcePort
from core.application.ports.vector_store import VectorStorePort
from core.config.settings import Settings
from core.infrastructure.persistence.chroma_vector_store import ChromaVectorStore
from core.infrastructure.persistence.faiss_vector_store import FaissVectorStore
from core.infrastructure.persistence.in_memory_section_store import InMemorySectionStore
from core.infrastructure.persistence.in_memory_vector_store import InMemoryVectorStore
from core.infrastructure.persistence.json_section_store import JsonSectionStore

VectorStoreFactory = Callable[[Settings], VectorStorePort]
SectionStoreFactory = Callable[[Settings], SectionSourcePort]


class VectorStoreRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, VectorStoreFactory] = {}

    def register(self, provider_id: str, factory: VectorStoreFactory) -> None:
        self._factories[provider_id] = factory

    def provider_ids(self) -> set[str]:
        return set(self._factories)

    def build(self, provider_id: str, settings: Settings) -> VectorStorePort:
        try:
            return self._factories[provider_id](settings)
        except KeyError as exc:
            raise ValueError(f"Unknown vector store provider: {provider_id}") from exc


vector_store_registry = VectorStoreRegistry()
vector_store_registry.register(
    "faiss",
    lambda settings: FaissVectorStore(
        dimension=settings.embedding_dimension,
        index_path=settings.faiss_index_path,
    ),
)
vector_store_registry.register(
    "memory",
    lambda _settings: InMemoryVectorStore(),
)
vector_store_registry.register(
    "chroma",
    lambda settings: ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
    ),
)


class SectionStoreRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, SectionStoreFactory] = {}

    def register(self, provider_id: str, factory: SectionStoreFactory) -> None:
        self._factories[provider_id] = factory

    def provider_ids(self) -> set[str]:
        return set(self._factories)

    def build(self, provider_id: str, settings: Settings) -> SectionSourcePort:
        try:
            return self._factories[provider_id](settings)
        except KeyError as exc:
            raise ValueError(f"Unknown section store provider: {provider_id}") from exc


section_store_registry = SectionStoreRegistry()
section_store_registry.register(
    "json",
    lambda settings: JsonSectionStore(path=settings.index_dir / "sections.json"),
)
section_store_registry.register("memory", lambda _settings: InMemorySectionStore())
