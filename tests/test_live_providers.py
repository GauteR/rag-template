import os

import pytest

from app.container import AppContainer
from core.config.settings import Settings


@pytest.mark.live_models
def test_live_ollama_embedding_dimension() -> None:
    if os.getenv("RUN_LIVE_MODELS") != "1":
        pytest.skip("Set RUN_LIVE_MODELS=1 to run live provider tests")

    settings = Settings(
        embedding_provider="ollama",
        embedding_dimension=int(os.getenv("LIVE_EMBEDDING_DIMENSION", "768")),
    )
    container = AppContainer(settings=settings)
    embedding = container.embedder.embed_texts(["live probe"])[0]
    assert len(embedding) == settings.embedding_dimension


@pytest.mark.live_models
def test_live_openai_embedding_dimension() -> None:
    if os.getenv("RUN_LIVE_MODELS") != "1":
        pytest.skip("Set RUN_LIVE_MODELS=1 to run live provider tests")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for live OpenAI tests")

    settings = Settings(
        embedding_provider="openai_compatible",
        embedding_dimension=int(os.getenv("LIVE_EMBEDDING_DIMENSION", "1536")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    container = AppContainer(settings=settings)
    embedding = container.embedder.embed_texts(["live probe"])[0]
    assert len(embedding) == settings.embedding_dimension
