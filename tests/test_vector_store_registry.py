import pytest

from core.config.settings import Settings
from core.infrastructure.persistence.registry import section_store_registry, vector_store_registry


def test_section_store_registry_includes_expected_providers() -> None:
    assert {"json", "memory"}.issubset(section_store_registry.provider_ids())


def test_section_store_registry_builds_memory_store() -> None:
    store = section_store_registry.build("memory", Settings())
    assert store.doc_ids() == set()


def test_section_store_registry_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown section store provider"):
        section_store_registry.build("missing", Settings())


def test_vector_store_registry_includes_expected_providers() -> None:
    assert {"faiss", "memory", "chroma"}.issubset(vector_store_registry.provider_ids())


def test_vector_store_registry_builds_memory_store() -> None:
    store = vector_store_registry.build("memory", Settings())
    assert store.count() == 0
    assert store.chunk_counts_by_doc() == {}


def test_vector_store_registry_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown vector store provider"):
        vector_store_registry.build("missing", Settings())
