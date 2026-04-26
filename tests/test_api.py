from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.container import AppContainer
from app.main import create_app
from core.config.settings import Settings


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_health_reports_configured_providers(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["llm_provider"] == "echo"
    assert response.json()["embedding_provider"] == "hash"


def test_index_markdown_and_query_api_flow(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    index_response = client.post(
        "/v1/index/markdown",
        json={"doc_id": "manual", "markdown": "# Intro\nWelcome\n\n## Install\nInstall with uv"},
    )
    query_response = client.post(
        "/v1/query",
        json={"question": "How do I install?", "k_recall": 10, "k_candidates": 5, "k_final": 1},
    )

    assert index_response.status_code == 200
    assert index_response.json()["indexed_chunks"] == 2
    assert query_response.status_code == 200
    body = query_response.json()
    assert body["sources"][0]["node_id"] == "manual:n2"
    assert "Install with uv" in body["answer"]


def test_query_remains_consistent_after_container_restart(tmp_path) -> None:
    first_container = AppContainer(settings=Settings(index_dir=tmp_path))
    first_client = TestClient(create_app(container=first_container))
    first_client.post(
        "/v1/index/markdown",
        json={"doc_id": "manual", "markdown": "# Intro\nWelcome\n\n## Install\nInstall with uv"},
    )

    restarted_container = AppContainer(settings=Settings(index_dir=tmp_path))
    restarted_client = TestClient(create_app(container=restarted_container))
    query_response = restarted_client.post(
        "/v1/query",
        json={"question": "How do I install?", "k_recall": 10, "k_candidates": 5, "k_final": 1},
    )

    assert query_response.status_code == 200
    assert query_response.json()["sources"][0]["node_id"] == "manual:n2"


def test_api_key_header_is_required_when_configured(tmp_path) -> None:
    container = AppContainer(settings=Settings(api_key="secret", index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    unauthorized = client.get("/v1/health")
    authorized = client.get("/v1/health", headers={"X-API-Key": "secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_index_pdf_is_enabled_by_default(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    response = client.post(
        "/v1/index/pdf",
        data={"doc_id": "manual-pdf"},
        files={"file": ("manual.pdf", _pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["doc_id"] == "manual-pdf"
    assert body["indexed_chunks"] >= 1


def test_index_pdf_uses_filename_as_default_doc_id(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    response = client.post(
        "/v1/index/pdf",
        files={"file": ("guide.pdf", _pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["doc_id"] == "guide"


def test_index_pdf_returns_404_when_llamaparse_disabled(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path, enable_llamaparse=False))
    client = TestClient(create_app(container=container))

    response = client.post(
        "/v1/index/pdf",
        data={"doc_id": "manual-pdf"},
        files={"file": ("manual.pdf", _pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 404


def test_index_pdf_returns_422_for_invalid_pdf_payload(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    response = client.post(
        "/v1/index/pdf",
        data={"doc_id": "manual-pdf"},
        files={"file": ("manual.pdf", b"not-a-valid-pdf", "application/pdf")},
    )

    assert response.status_code == 422
