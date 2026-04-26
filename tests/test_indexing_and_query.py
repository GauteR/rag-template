import pytest

from core.application.indexing.chunking import StructureGuidedChunker
from core.application.indexing.markdown_parser import MarkdownSkeletonParser
from core.application.indexing.noise_filter import HeuristicNoiseFilter
from core.application.indexing.use_case import IndexMarkdownUseCase
from core.application.ports.embeddings import EmbedderPort
from core.application.ports.llm import LlmPort
from core.application.query.use_case import QueryUseCase
from core.domain.models import Section
from core.infrastructure.persistence.in_memory_section_store import (
    InMemorySectionStore,
)
from core.infrastructure.persistence.in_memory_vector_store import (
    InMemoryVectorStore,
)


class KeywordEmbedder(EmbedderPort):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if "install" in text.lower() else 0.0] for text in texts]


class PartialReranker(LlmPort):
    def filter_noise(self, *, sections: list[Section]) -> set[str]:
        del sections
        return set()

    def rerank(self, *, question: str, candidates: list[Section], k_final: int) -> list[str]:
        del question, candidates, k_final
        return ["missing-node"]

    def synthesize(self, *, question: str, sections: list[Section]) -> str:
        del question
        return "\n".join(section.text for section in sections)


def test_index_markdown_stores_vectors_and_full_sections() -> None:
    vector_store = InMemoryVectorStore()
    section_store = InMemorySectionStore()
    use_case = IndexMarkdownUseCase(
        parser=MarkdownSkeletonParser(),
        chunker=StructureGuidedChunker(),
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
        section_source=section_store,
    )

    result = use_case.execute(
        doc_id="manual",
        markdown="# Intro\nWelcome\n\n## Install\nInstall with uv",
    )

    assert result.indexed_chunks == 2
    assert vector_store.count() == 2
    assert section_store.get_section("manual", "manual:n2").text == "## Install\nInstall with uv"


def test_reindexing_replaces_previous_vectors_and_sections_for_same_doc() -> None:
    vector_store = InMemoryVectorStore()
    section_store = InMemorySectionStore()
    use_case = IndexMarkdownUseCase(
        parser=MarkdownSkeletonParser(),
        chunker=StructureGuidedChunker(),
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
        section_source=section_store,
    )
    use_case.execute(
        doc_id="manual",
        markdown="# Intro\nWelcome\n\n## Install\nInstall with uv",
    )

    result = use_case.execute(doc_id="manual", markdown="# Query\nAsk questions")

    assert result.indexed_sections == 1
    assert vector_store.count() == 1
    assert section_store.get_section("manual", "manual:n1").text == "# Query\nAsk questions"
    with pytest.raises(KeyError):
        section_store.get_section("manual", "manual:n2")


def test_section_store_pointer_fetch_includes_child_sections() -> None:
    section_store = InMemorySectionStore()
    document = MarkdownSkeletonParser().parse(
        doc_id="manual",
        markdown="# Parent\nParent text\n\n## Child\nChild text",
    )

    section_store.store_document(document)

    parent_section = section_store.get_section("manual", "manual:n1")
    assert "Parent text" in parent_section.text
    assert "## Child\nChild text" in parent_section.text


def test_index_markdown_filters_heuristic_noise_sections() -> None:
    vector_store = InMemoryVectorStore()
    section_store = InMemorySectionStore()
    use_case = IndexMarkdownUseCase(
        parser=MarkdownSkeletonParser(),
        chunker=StructureGuidedChunker(),
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
        section_source=section_store,
        noise_filter=HeuristicNoiseFilter(noisy_titles={"disclaimer"}),
    )

    result = use_case.execute(
        doc_id="manual",
        markdown="# Useful\nInstall with uv\n\n# Disclaimer\nIgnore this legal boilerplate",
    )

    assert result.indexed_sections == 1
    assert vector_store.count() == 1


def test_query_uses_broad_recall_dedupes_nodes_and_fetches_full_sections() -> None:
    vector_store = InMemoryVectorStore()
    section_store = InMemorySectionStore()
    index_use_case = IndexMarkdownUseCase(
        parser=MarkdownSkeletonParser(),
        chunker=StructureGuidedChunker(),
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
        section_source=section_store,
    )
    index_use_case.execute(
        doc_id="manual",
        markdown="# Intro\nWelcome\n\n## Install\nInstall with uv\n\n## Query\nAsk questions",
    )
    query_use_case = QueryUseCase(
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
        section_source=section_store,
        synthesis_llm=None,
        reranker_llm=None,
        enable_llm_reranker=False,
    )

    response = query_use_case.execute(
        question="How do I install?", k_recall=10, k_candidates=5, k_final=1
    )

    assert response.sources[0].node_id == "manual:n2"
    assert response.sources[0].breadcrumb == ("Intro", "Install")
    assert "Install with uv" in response.answer
    assert response.sources[0].citation == "## Install"
    assert response.sources[0].start_offset is not None
    assert response.sources[0].end_offset is not None
    assert response.sources[0].start_offset < response.sources[0].end_offset


def test_query_sources_keep_vector_scores() -> None:
    vector_store = InMemoryVectorStore()
    section_store = InMemorySectionStore()
    index_use_case = IndexMarkdownUseCase(
        parser=MarkdownSkeletonParser(),
        chunker=StructureGuidedChunker(),
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
        section_source=section_store,
    )
    index_use_case.execute(
        doc_id="manual",
        markdown="# Intro\nWelcome\n\n## Install\nInstall with uv",
    )
    query_use_case = QueryUseCase(
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
        section_source=section_store,
        synthesis_llm=None,
        reranker_llm=None,
        enable_llm_reranker=False,
    )

    response = query_use_case.execute(
        question="How do I install?",
        k_recall=10,
        k_candidates=5,
        k_final=2,
    )

    assert response.sources[0].score > response.sources[1].score


def test_query_reranker_falls_back_to_embedding_order_when_ids_are_invalid() -> None:
    vector_store = InMemoryVectorStore()
    section_store = InMemorySectionStore()
    index_use_case = IndexMarkdownUseCase(
        parser=MarkdownSkeletonParser(),
        chunker=StructureGuidedChunker(),
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
        section_source=section_store,
    )
    index_use_case.execute(
        doc_id="manual",
        markdown="# Intro\nWelcome\n\n## Install\nInstall with uv",
    )
    query_use_case = QueryUseCase(
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
        section_source=section_store,
        synthesis_llm=None,
        reranker_llm=PartialReranker(),
        enable_llm_reranker=True,
    )

    response = query_use_case.execute(
        question="How do I install?",
        k_recall=10,
        k_candidates=5,
        k_final=1,
    )

    assert response.sources[0].node_id == "manual:n2"


def test_section_store_computes_citation_and_offsets_for_markdown_sections() -> None:
    section_store = InMemorySectionStore()
    markdown = "# Intro\nWelcome\n\n## Install\nInstall with uv"
    document = MarkdownSkeletonParser().parse(doc_id="manual", markdown=markdown)

    section_store.store_document(document)

    install_section = section_store.get_section("manual", "manual:n2")
    assert install_section.citation == "## Install"
    assert install_section.start_offset is not None
    assert install_section.end_offset is not None
    assert install_section.start_offset < install_section.end_offset
    assert markdown[install_section.start_offset : install_section.end_offset].startswith(
        "## Install"
    )


def test_section_store_offsets_cover_child_sections() -> None:
    section_store = InMemorySectionStore()
    markdown = "# Parent\nParent text\n\n## Child\nChild text"
    document = MarkdownSkeletonParser().parse(doc_id="manual", markdown=markdown)

    section_store.store_document(document)

    parent_section = section_store.get_section("manual", "manual:n1")
    child_section = section_store.get_section("manual", "manual:n2")
    assert parent_section.start_offset == 0
    assert parent_section.end_offset == len(markdown)
    assert parent_section.end_offset >= child_section.end_offset


def test_section_without_offset_metadata_serializes_successfully() -> None:
    section = Section(
        doc_id="doc",
        node_id="doc:n1",
        breadcrumb=("Title",),
        text="Some text",
    )
    assert section.citation is None
    assert section.start_offset is None
    assert section.end_offset is None
