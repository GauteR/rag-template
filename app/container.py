from __future__ import annotations

from functools import cached_property

from rag_template_core.application.indexing.chunking import StructureGuidedChunker
from rag_template_core.application.indexing.markdown_parser import MarkdownSkeletonParser
from rag_template_core.application.indexing.noise_filter import HeuristicNoiseFilter, LlmNoiseFilter
from rag_template_core.application.indexing.use_case import IndexMarkdownUseCase
from rag_template_core.application.query.use_case import QueryUseCase
from rag_template_core.config.settings import Settings
from rag_template_core.infrastructure.embeddings.registry import embedding_registry
from rag_template_core.infrastructure.llm.registry import llm_registry
from rag_template_core.infrastructure.persistence.faiss_vector_store import FaissVectorStore
from rag_template_core.infrastructure.persistence.json_section_store import JsonSectionStore


class AppContainer:
    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.validate_provider_ids(
            llm_provider_ids=llm_registry.provider_ids(),
            embedding_provider_ids=embedding_registry.provider_ids(),
        )
        self.settings.validate_provider_configuration()

    @cached_property
    def embedder(self):
        return embedding_registry.build(self.settings.embedding_provider, self.settings)

    @cached_property
    def synthesis_llm(self):
        return llm_registry.build(self.settings.synthesis_provider, self.settings)

    @cached_property
    def reranker_llm(self):
        return llm_registry.build(self.settings.routing_provider, self.settings)

    @cached_property
    def vector_store(self) -> FaissVectorStore:
        return FaissVectorStore(dimension=self.settings.embedding_dimension)

    @cached_property
    def section_store(self) -> JsonSectionStore:
        return JsonSectionStore(path=self.settings.index_dir / "sections.json")

    def index_markdown_use_case(self) -> IndexMarkdownUseCase:
        return IndexMarkdownUseCase(
            parser=MarkdownSkeletonParser(),
            chunker=StructureGuidedChunker(),
            embedder=self.embedder,
            vector_store=self.vector_store,
            section_source=self.section_store,
            noise_filter=self._noise_filter(),
        )

    def _noise_filter(self):
        if self.settings.enable_llm_noise_filter:
            return LlmNoiseFilter(llm=self.reranker_llm)
        return HeuristicNoiseFilter()

    def query_use_case(self) -> QueryUseCase:
        return QueryUseCase(
            embedder=self.embedder,
            vector_store=self.vector_store,
            section_source=self.section_store,
            synthesis_llm=self.synthesis_llm,
            reranker_llm=self.reranker_llm,
            enable_llm_reranker=self.settings.enable_llm_reranker,
        )
