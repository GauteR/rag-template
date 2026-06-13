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


def test_streaming_query_does_not_call_synthesize_on_llm() -> None:
    from core.application.indexing.chunking import StructureGuidedChunker
    from core.application.indexing.markdown_parser import MarkdownSkeletonParser
    from core.application.indexing.use_case import IndexMarkdownUseCase
    from core.application.ports.llm import LlmPort
    from core.application.query.use_case import QueryUseCase
    from core.domain.models import Section
    from core.infrastructure.embeddings.providers.hash import HashEmbedder
    from core.infrastructure.persistence.in_memory_section_store import InMemorySectionStore
    from core.infrastructure.persistence.in_memory_vector_store import InMemoryVectorStore

    class CountingLlm(LlmPort):
        def __init__(self) -> None:
            self.synthesize_count = 0
            self.synthesize_stream_count = 0

        def filter_noise(self, *, sections: list[Section]) -> set[str]:
            return set()

        def rerank(self, *, question: str, candidates: list[Section], k_final: int) -> list[str]:
            del question, candidates, k_final
            return []

        def synthesize(self, *, question: str, sections: list[Section]) -> str:
            del question, sections
            self.synthesize_count += 1
            return "blocking answer"

        def synthesize_stream(self, *, question: str, sections: list[Section]):
            del question, sections
            self.synthesize_stream_count += 1
            yield "streamed answer"

    vector_store = InMemoryVectorStore()
    section_store = InMemorySectionStore()
    IndexMarkdownUseCase(
        parser=MarkdownSkeletonParser(),
        chunker=StructureGuidedChunker(),
        embedder=HashEmbedder(dimension=8),
        vector_store=vector_store,
        section_source=section_store,
    ).execute(doc_id="demo", markdown="# Demo\nInstall with uv")

    llm = CountingLlm()
    use_case = QueryUseCase(
        embedder=HashEmbedder(dimension=8),
        vector_store=vector_store,
        section_source=section_store,
        synthesis_llm=llm,
        reranker_llm=llm,
        enable_llm_reranker=False,
    )
    sources, token_stream = use_case.synthesize_stream(
        question="How do I install?",
        k_recall=5,
        k_candidates=3,
        k_final=1,
    )

    assert list(token_stream) == ["streamed answer"]
    assert sources
    assert llm.synthesize_count == 0
    assert llm.synthesize_stream_count == 1
