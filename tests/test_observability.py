from fastapi.testclient import TestClient

from app.container import AppContainer
from app.main import create_app
from core.config.settings import Settings


def test_query_tracing_metadata(tmp_path) -> None:
    container = AppContainer(
        settings=Settings(index_dir=tmp_path, enable_query_tracing=True),
    )
    client = TestClient(create_app(container=container))

    client.post(
        "/v1/index/markdown",
        json={"doc_id": "demo", "markdown": "# Demo\nInstall with uv"},
    )
    response = client.post(
        "/v1/query",
        json={"question": "How do I install?", "k_recall": 5, "k_candidates": 3, "k_final": 1},
    )

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata is not None
    assert metadata["recalled_count"] >= 1
    assert metadata["candidate_count"] >= 1
    assert metadata["final_count"] >= 1


def test_streaming_query_disabled_by_default(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    response = client.post(
        "/v1/query/stream",
        json={"question": "Hello?", "k_recall": 5, "k_candidates": 3, "k_final": 1},
    )
    assert response.status_code == 404


def test_streaming_query_returns_sse_events(tmp_path) -> None:
    container = AppContainer(
        settings=Settings(index_dir=tmp_path, enable_streaming_query=True),
    )
    client = TestClient(create_app(container=container))

    client.post(
        "/v1/index/markdown",
        json={"doc_id": "demo", "markdown": "# Demo\nInstall with uv"},
    )
    response = client.post(
        "/v1/query/stream",
        json={"question": "How do I install?", "k_recall": 5, "k_candidates": 3, "k_final": 1},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: sources" in body
    assert "event: token" in body
    assert "event: done" in body
