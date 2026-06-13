from __future__ import annotations

import logging
from functools import cached_property

from core.application.indexing.chunking import StructureGuidedChunker
from core.application.indexing.markdown_parser import MarkdownSkeletonParser
from core.application.indexing.noise_filter import HeuristicNoiseFilter, LlmNoiseFilter
from core.application.indexing.use_case import IndexMarkdownUseCase
from core.application.ports.vector_store import VectorStorePort
from core.application.query.use_case import QueryUseCase
from core.config.settings import Settings
from core.infrastructure.embeddings.registry import embedding_registry
from core.infrastructure.extraction.llamaparse_pdf_extractor import LlamaParsePdfExtractor
from core.infrastructure.llm.registry import llm_registry
from core.infrastructure.persistence.json_section_store import JsonSectionStore
from core.infrastructure.persistence.registry import vector_store_registry

logger = logging.getLogger(__name__)


class AppContainer:
    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._embedding_probe_error: str | None = None
        self._probe_embedding_dimension()

    def _probe_embedding_dimension(self) -> None:
        try:
            embedding = self.embedder.embed_texts(["dimension probe"])[0]
            self.settings.validate_embedding_dimension(actual_dimension=len(embedding))
        except Exception as exc:
            self._embedding_probe_error = str(exc)
            logger.warning("Embedding dimension probe failed: %s", exc)

    def collect_config_errors(self) -> list[str]:
        """Return a list of configuration error messages without raising or exposing secrets."""
        errors: list[str] = []
        try:
            self.settings.validate_provider_ids(
                llm_provider_ids=llm_registry.provider_ids(),
                embedding_provider_ids=embedding_registry.provider_ids(),
                vector_store_provider_ids=vector_store_registry.provider_ids(),
            )
        except ValueError as exc:
            errors.append(str(exc))
        try:
            self.settings.validate_provider_configuration()
        except ValueError as exc:
            errors.append(str(exc))
        if self._embedding_probe_error is not None:
            errors.append(self._embedding_probe_error)
        return errors

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
    def vector_store(self) -> VectorStorePort:
        return vector_store_registry.build(self.settings.vector_store_provider, self.settings)

    @cached_property
    def section_store(self) -> JsonSectionStore:
        return JsonSectionStore(path=self.settings.index_dir / "sections.json")

    @cached_property
    def pdf_extractor(self) -> LlamaParsePdfExtractor:
        return LlamaParsePdfExtractor(api_key=self.settings.llama_cloud_api_key)

    def index_markdown_use_case(self) -> IndexMarkdownUseCase:
        lexical_store = None
        if self.settings.enable_hybrid_search:
            from core.infrastructure.persistence.bm25_lexical_store import Bm25LexicalStore

            lexical_store = Bm25LexicalStore(path=self.settings.index_dir / "lexical.json")

        return IndexMarkdownUseCase(
            parser=MarkdownSkeletonParser(),
            chunker=StructureGuidedChunker(),
            embedder=self.embedder,
            vector_store=self.vector_store,
            section_source=self.section_store,
            noise_filter=self._noise_filter(),
            lexical_store=lexical_store,
        )

    def _noise_filter(self):
        if self.settings.enable_llm_noise_filter:
            return LlmNoiseFilter(llm=self.reranker_llm)
        return HeuristicNoiseFilter()

    def query_use_case(self) -> QueryUseCase:
        lexical_store = None
        if self.settings.enable_hybrid_search:
            from core.infrastructure.persistence.bm25_lexical_store import Bm25LexicalStore

            lexical_store = Bm25LexicalStore(path=self.settings.index_dir / "lexical.json")

        return QueryUseCase(
            embedder=self.embedder,
            vector_store=self.vector_store,
            section_source=self.section_store,
            synthesis_llm=self.synthesis_llm,
            reranker_llm=self.reranker_llm,
            enable_llm_reranker=self.settings.enable_llm_reranker,
            enable_query_tracing=self.settings.enable_query_tracing,
            enable_hybrid_search=self.settings.enable_hybrid_search,
            lexical_store=lexical_store,
        )

    def faiss_available(self) -> bool:
        from core.infrastructure.persistence.faiss_vector_store import FaissVectorStore

        if isinstance(self.vector_store, FaissVectorStore):
            return self.vector_store.is_faiss_loaded()
        return False

    def chroma_reachable(self) -> bool | None:
        if self.settings.vector_store_provider != "chroma":
            return None
        try:
            self.vector_store.count()
            return True
        except Exception:
            return False
