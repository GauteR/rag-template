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
    body = response.json()
    assert body["llm_provider"] == "echo"
    assert body["routing_provider"] == "echo"
    assert body["embedding_provider"] == "hash"
    assert body["status"] == "ok"
    assert body["config_errors"] == []


def test_health_reports_degraded_for_unknown_llm_provider(tmp_path) -> None:
    container = AppContainer(settings=Settings(llm_provider="no_such_provider", index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert any("no_such_provider" in err for err in body["config_errors"])


def test_health_reports_degraded_for_missing_required_config(tmp_path) -> None:
    container = AppContainer(
        settings=Settings(llm_provider="anthropic", embedding_provider="hash", index_dir=tmp_path)
    )
    client = TestClient(create_app(container=container))

    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert any("ANTHROPIC_API_KEY" in err for err in body["config_errors"])
    assert not any(
        secret in str(body) for secret in ["sk-", "anthropic-", "Bearer"]
    ), "Health response must not expose secret values"


def test_health_reports_empty_index(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["index_ready"] is False
    assert body["index_document_count"] == 0
    assert body["index_consistent"] is True


def test_health_reports_ready_index_after_indexing(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    client.post(
        "/v1/index/markdown",
        json={"doc_id": "doc1", "markdown": "# Hello\nWorld"},
    )
    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["index_ready"] is True
    assert body["index_document_count"] == 1
    assert body["index_consistent"] is True


def test_health_reports_inconsistent_index(tmp_path) -> None:
    from core.domain.models import VectorRecord

    container = AppContainer(settings=Settings(index_dir=tmp_path))
    # Manually add a vector record without a corresponding section store entry
    container.vector_store.add(
        [
            VectorRecord(
                doc_id="orphan",
                node_id="n1",
                chunk_id="orphan:n1:c0",
                embedding=(0.0,) * 8,
                text="orphan chunk",
                breadcrumb=("orphan",),
            )
        ]
    )
    client = TestClient(create_app(container=container))

    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["index_consistent"] is False
    assert body["index_document_count"] == 1


def test_health_returns_degraded_when_index_inspection_raises(tmp_path) -> None:
    container = AppContainer(settings=Settings(index_dir=tmp_path))
    client = TestClient(create_app(container=container))

    # Corrupt the section store's backing file so doc_ids() raises
    sections_path = tmp_path / "sections.json"
    sections_path.write_text("not-valid-json", encoding="utf-8")
    # Force a fresh section_store that reads the corrupt file
    container.__dict__.pop("section_store", None)

    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["index_ready"] is False
    assert any("Index inspection failed" in err for err in body["config_errors"])


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
    source = body["sources"][0]
    assert source["citation"] == "## Install"
    assert source["start_offset"] is not None
    assert source["end_offset"] is not None
    assert source["start_offset"] < source["end_offset"]


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
