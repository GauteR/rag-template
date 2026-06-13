import pytest
from fastapi.testclient import TestClient

from app.container import AppContainer
from app.main import create_app
from core.config.settings import Settings


def test_index_admin_disabled_by_default(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    assert client.get("/v1/index").status_code == 404
    assert client.delete("/v1/index/demo").status_code == 404


def test_index_admin_list_and_delete(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path, enable_index_admin=True))
    client = TestClient(create_app(container=container))

    client.post(
        "/v1/index/markdown",
        json={"doc_id": "demo", "markdown": "# Demo\nHello"},
    )

    list_response = client.get("/v1/index")
    assert list_response.status_code == 200
    body = list_response.json()
    assert body["documents"][0]["doc_id"] == "demo"
    assert body["documents"][0]["section_count"] == 1
    assert body["documents"][0]["chunk_count"] == 1

    delete_response = client.delete("/v1/index/demo")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    query_response = client.post(
        "/v1/query",
        json={"question": "Hello?", "k_recall": 5, "k_candidates": 3, "k_final": 1},
    )
    assert query_response.status_code == 200
    assert query_response.json()["sources"] == []


def test_index_admin_delete_removes_hybrid_lexical_index(tmp_path) -> None:
    pytest.importorskip("rank_bm25")

    container = AppContainer(
        settings=Settings(
            index_dir=tmp_path,
            enable_index_admin=True,
            enable_hybrid_search=True,
        )
    )
    client = TestClient(create_app(container=container))

    client.post(
        "/v1/index/markdown",
        json={"doc_id": "demo", "markdown": "# Demo\nInstall with uv sync"},
    )
    lexical_path = tmp_path / "lexical.json"
    assert lexical_path.exists()

    delete_response = client.delete("/v1/index/demo")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert lexical_path.exists() is False or "demo" not in lexical_path.read_text(encoding="utf-8")
