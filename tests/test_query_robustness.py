from core.application.indexing.chunking import StructureGuidedChunker
from core.application.indexing.markdown_parser import MarkdownSkeletonParser
from core.application.indexing.use_case import IndexMarkdownUseCase
from core.application.query.use_case import QueryUseCase
from core.domain.models import VectorRecord
from core.infrastructure.embeddings.providers.hash import HashEmbedder
from core.infrastructure.llm.providers.echo import EchoLlm
from core.infrastructure.persistence.in_memory_section_store import InMemorySectionStore
from core.infrastructure.persistence.in_memory_vector_store import InMemoryVectorStore


def test_query_skips_orphan_vector_hits() -> None:
    vector_store = InMemoryVectorStore()
    section_store = InMemorySectionStore()
    index_use_case = IndexMarkdownUseCase(
        parser=MarkdownSkeletonParser(),
        chunker=StructureGuidedChunker(),
        embedder=HashEmbedder(dimension=8),
        vector_store=vector_store,
        section_source=section_store,
    )
    index_use_case.execute(
        doc_id="manual",
        markdown="# Intro\nWelcome\n\n## Install\nInstall with uv",
    )

    vector_store.add(
        [
            VectorRecord(
                doc_id="manual",
                node_id="manual:orphan",
                chunk_id="manual:orphan:c1",
                embedding=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                text="orphan chunk",
                breadcrumb=("Orphan",),
            )
        ]
    )

    response = QueryUseCase(
        embedder=HashEmbedder(dimension=8),
        vector_store=vector_store,
        section_source=section_store,
        synthesis_llm=EchoLlm(),
        reranker_llm=EchoLlm(),
        enable_llm_reranker=False,
    ).execute(question="How do I install?", k_recall=10, k_candidates=5, k_final=2)

    node_ids = {source.node_id for source in response.sources}
    assert "manual:orphan" not in node_ids
    assert "manual:n2" in node_ids
