from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

from core.application.ports.embeddings import EmbedderPort
from core.application.ports.lexical_store import LexicalStorePort
from core.application.ports.llm import LlmPort
from core.application.ports.section_source import SectionSourcePort
from core.application.ports.vector_store import VectorStorePort
from core.application.query.models import QueryMetadata, QueryResponse, QuerySource
from core.domain.models import SearchHit, Section

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CandidateHit:
    hit: SearchHit
    section: Section


@dataclass(frozen=True)
class _PreparedQuery:
    sources: list[QuerySource]
    sections: list[Section]
    metadata: QueryMetadata | None


class QueryUseCase:
    def __init__(
        self,
        *,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        section_source: SectionSourcePort,
        synthesis_llm: LlmPort | None,
        reranker_llm: LlmPort | None,
        enable_llm_reranker: bool,
        enable_query_tracing: bool = False,
        enable_hybrid_search: bool = False,
        lexical_store: LexicalStorePort | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._section_source = section_source
        self._synthesis_llm = synthesis_llm
        self._reranker_llm = reranker_llm
        self._enable_llm_reranker = enable_llm_reranker
        self._enable_query_tracing = enable_query_tracing
        self._enable_hybrid_search = enable_hybrid_search
        self._lexical_store = lexical_store

    def execute(
        self,
        *,
        question: str,
        k_recall: int = 200,
        k_candidates: int = 50,
        k_final: int = 5,
        doc_id: str | None = None,
        min_score: float | None = None,
    ) -> QueryResponse:
        prepared = self._prepare_query(
            question=question,
            k_recall=k_recall,
            k_candidates=k_candidates,
            k_final=k_final,
            doc_id=doc_id,
            min_score=min_score,
        )

        synth_start = perf_counter()
        answer = self._synthesize(question=question, sections=prepared.sections)
        metadata = prepared.metadata
        if metadata is not None:
            metadata.synthesize_ms = (perf_counter() - synth_start) * 1_000

        return QueryResponse(answer=answer, sources=prepared.sources, metadata=metadata)

    def synthesize_stream(
        self,
        *,
        question: str,
        k_recall: int = 200,
        k_candidates: int = 50,
        k_final: int = 5,
        doc_id: str | None = None,
        min_score: float | None = None,
    ) -> tuple[list[QuerySource], Iterator[str]]:
        prepared = self._prepare_query(
            question=question,
            k_recall=k_recall,
            k_candidates=k_candidates,
            k_final=k_final,
            doc_id=doc_id,
            min_score=min_score,
        )

        if self._synthesis_llm is not None:
            stream = self._synthesis_llm.synthesize_stream(
                question=question,
                sections=prepared.sections,
            )
        else:
            stream = iter([self._synthesize(question=question, sections=prepared.sections)])

        return prepared.sources, stream

    def _prepare_query(
        self,
        *,
        question: str,
        k_recall: int,
        k_candidates: int,
        k_final: int,
        doc_id: str | None,
        min_score: float | None,
    ) -> _PreparedQuery:
        metadata = QueryMetadata() if self._enable_query_tracing else None
        start = perf_counter()

        query_embedding = self._embedder.embed_texts([question])[0]
        recalled = self._vector_store.search(query_embedding, limit=k_recall)
        if self._enable_hybrid_search and self._lexical_store is not None:
            lexical_hits = self._lexical_store.search(
                query=question,
                limit=k_recall,
                doc_id=doc_id,
            )
            recalled = self._merge_hits(recalled, lexical_hits)
        if doc_id is not None:
            recalled = [hit for hit in recalled if hit.record.doc_id == doc_id]
        if min_score is not None:
            recalled = [hit for hit in recalled if hit.score >= min_score]

        if metadata is not None:
            metadata.recall_ms = (perf_counter() - start) * 1_000
            metadata.recalled_count = len(recalled)

        candidates = self._dedupe_nodes(recalled)[:k_candidates]
        resolved = self._resolve_sections(candidates)
        scores_by_node_id = {item.hit.record.node_id: item.hit.score for item in resolved}

        rerank_start = perf_counter()
        final_sections = self._rank_sections(
            question=question,
            sections=[item.section for item in resolved],
            k_final=k_final,
        )
        if metadata is not None:
            metadata.rerank_ms = (perf_counter() - rerank_start) * 1_000
            metadata.candidate_count = len(resolved)
            metadata.final_count = len(final_sections)
            metadata.enable_llm_reranker = self._enable_llm_reranker
            metadata.enable_hybrid_search = self._enable_hybrid_search

        sources = [
            QuerySource(
                doc_id=section.doc_id,
                node_id=section.node_id,
                breadcrumb=section.breadcrumb,
                score=scores_by_node_id.get(section.node_id, 0.0),
                text=section.text,
                citation=section.citation,
                start_offset=section.start_offset,
                end_offset=section.end_offset,
            )
            for section in final_sections
        ]

        return _PreparedQuery(sources=sources, sections=final_sections, metadata=metadata)

    def _resolve_sections(self, candidates: list[SearchHit]) -> list[_CandidateHit]:
        resolved: list[_CandidateHit] = []
        for hit in candidates:
            try:
                section = self._section_source.get_section(hit.record.doc_id, hit.record.node_id)
            except KeyError:
                logger.warning(
                    "Orphan vector hit skipped: doc_id=%s node_id=%s",
                    hit.record.doc_id,
                    hit.record.node_id,
                )
                continue
            resolved.append(_CandidateHit(hit=hit, section=section))
        return resolved

    def _merge_hits(
        self,
        vector_hits: list[SearchHit],
        lexical_hits: list[SearchHit],
    ) -> list[SearchHit]:
        # Vector cosine scores and BM25 scores use different scales; keep the higher raw score.
        merged: dict[tuple[str, str], SearchHit] = {}
        for hit in vector_hits + lexical_hits:
            key = (hit.record.doc_id, hit.record.node_id)
            existing = merged.get(key)
            if existing is None or hit.score > existing.score:
                merged[key] = hit
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)

    def _dedupe_nodes(self, hits: list[SearchHit]) -> list[SearchHit]:
        seen: set[tuple[str, str]] = set()
        unique: list[SearchHit] = []
        for hit in hits:
            record = hit.record
            key = (record.doc_id, record.node_id)
            if key not in seen:
                seen.add(key)
                unique.append(hit)
        return unique

    def _rank_sections(
        self, *, question: str, sections: list[Section], k_final: int
    ) -> list[Section]:
        if not self._enable_llm_reranker or self._reranker_llm is None:
            return sections[:k_final]

        ranked_node_ids = self._reranker_llm.rerank(
            question=question,
            candidates=sections,
            k_final=k_final,
        )
        sections_by_id = {section.node_id: section for section in sections}
        ranked = [
            sections_by_id[node_id] for node_id in ranked_node_ids if node_id in sections_by_id
        ]
        ranked_node_id_set = {section.node_id for section in ranked}
        ranked.extend(section for section in sections if section.node_id not in ranked_node_id_set)
        return ranked[:k_final]

    def _synthesize(self, *, question: str, sections: list[Section]) -> str:
        if self._synthesis_llm is not None:
            return self._synthesis_llm.synthesize(question=question, sections=sections)
        if not sections:
            return "No relevant sections found."
        context = "\n\n".join(section.text for section in sections)
        return f"Question: {question}\n\nContext:\n{context}"
