from __future__ import annotations

from core.application.indexing.chunking import StructureGuidedChunker
from core.application.indexing.markdown_parser import MarkdownSkeletonParser
from core.application.indexing.models import IndexMarkdownResult
from core.application.indexing.noise_filter import NoiseFilter
from core.application.ports.embeddings import EmbedderPort
from core.application.ports.section_source import SectionSourcePort
from core.application.ports.vector_store import VectorStorePort
from core.domain.models import VectorRecord


class IndexMarkdownUseCase:
    def __init__(
        self,
        *,
        parser: MarkdownSkeletonParser,
        chunker: StructureGuidedChunker,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        section_source: SectionSourcePort,
        noise_filter: NoiseFilter | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._section_source = section_source
        self._noise_filter = noise_filter

    def execute(self, *, doc_id: str, markdown: str) -> IndexMarkdownResult:
        document = self._parser.parse(doc_id=doc_id, markdown=markdown)
        if self._noise_filter is not None:
            document = self._noise_filter.filter(document)
        chunks = self._chunker.chunk(document)
        embeddings = self._embedder.embed_texts([chunk.embedding_text for chunk in chunks])

        records = [
            VectorRecord(
                doc_id=chunk.doc_id,
                node_id=chunk.node_id,
                chunk_id=chunk.chunk_id,
                embedding=tuple(embedding),
                text=chunk.text,
                breadcrumb=chunk.breadcrumb,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._section_source.delete_document(doc_id)
        self._vector_store.delete_document(doc_id)
        self._section_source.store_document(document)
        self._vector_store.add(records)

        return IndexMarkdownResult(
            doc_id=doc_id,
            indexed_chunks=len(records),
            indexed_sections=len(document.nodes),
        )
