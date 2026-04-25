from pathlib import Path


def test_dockerfile_runs_fastapi_app() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "uvicorn" in dockerfile
    assert "app.main:app" in dockerfile
    assert "EXPOSE 8000" in dockerfile


def test_docker_compose_defines_api_and_chroma_services() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "rag-api:" in compose
    assert "chroma:" in compose
    assert "8000:8000" in compose
    assert "8001:8000" in compose
