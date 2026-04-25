import pytest

from core.config.settings import Settings
from core.infrastructure.embeddings.registry import embedding_registry
from core.infrastructure.llm.registry import llm_registry


def test_default_registries_include_local_and_http_provider_ids() -> None:
    assert {"ollama", "openai_compatible", "hash"}.issubset(embedding_registry.provider_ids())
    assert {"ollama", "openai_compatible", "anthropic", "echo"}.issubset(
        llm_registry.provider_ids()
    )


def test_settings_validate_selected_provider_ids() -> None:
    settings = Settings(
        llm_provider="echo",
        embedding_provider="hash",
        embedding_dimension=8,
    )

    settings.validate_provider_ids(
        llm_provider_ids=llm_registry.provider_ids(),
        embedding_provider_ids=embedding_registry.provider_ids(),
    )


def test_settings_reject_unknown_provider_ids() -> None:
    settings = Settings(llm_provider="missing", embedding_provider="hash")

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        settings.validate_provider_ids(
            llm_provider_ids=llm_registry.provider_ids(),
            embedding_provider_ids=embedding_registry.provider_ids(),
        )


def test_settings_validate_required_active_provider_configuration() -> None:
    settings = Settings(llm_provider="anthropic", embedding_provider="hash")

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        settings.validate_provider_configuration()


def test_settings_require_openai_api_key_for_openai_compatible_providers() -> None:
    llm_settings = Settings(llm_provider="openai_compatible", embedding_provider="hash")
    embedding_settings = Settings(llm_provider="echo", embedding_provider="openai_compatible")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        llm_settings.validate_provider_configuration()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        embedding_settings.validate_provider_configuration()
