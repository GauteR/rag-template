from app.container import AppContainer
from core.config.settings import Settings


def test_container_reuses_cached_use_cases_and_lexical_store(tmp_path) -> None:
    settings = Settings(index_dir=tmp_path, enable_hybrid_search=True)
    container = AppContainer(settings=settings)

    first_index_use_case = container.index_markdown_use_case
    second_index_use_case = container.index_markdown_use_case
    first_query_use_case = container.query_use_case
    second_query_use_case = container.query_use_case
    first_lexical_store = container.lexical_store
    second_lexical_store = container.lexical_store

    assert first_index_use_case is second_index_use_case
    assert first_query_use_case is second_query_use_case
    assert first_lexical_store is second_lexical_store


def test_container_reports_faiss_missing_when_not_loaded(tmp_path, monkeypatch) -> None:
    from core.infrastructure.persistence.faiss_vector_store import FaissVectorStore

    monkeypatch.setattr(FaissVectorStore, "is_faiss_loaded", lambda _self: False)
    container = AppContainer(settings=Settings(index_dir=tmp_path, vector_store_provider="faiss"))

    errors = container.collect_config_errors()

    assert any("FAISS is not loaded" in error for error in errors)
