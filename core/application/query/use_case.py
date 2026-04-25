from __future__ import annotations

from core.application.ports.embeddings import EmbedderPort
from core.application.ports.llm import LlmPort
from core.application.ports.section_source import SectionSourcePort
from core.application.ports.vector_store import VectorStorePort
from core.application.query.models import QueryResponse, QuerySource
from core.domain.models import SearchHit, Section


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
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._section_source = section_source
        self._synthesis_llm = synthesis_llm
        self._reranker_llm = reranker_llm
        self._enable_llm_reranker = enable_llm_reranker

    def execute(
        self,
        *,
        question: str,
        k_recall: int = 200,
        k_candidates: int = 50,
        k_final: int = 5,
    ) -> QueryResponse:
        query_embedding = self._embedder.embed_texts([question])[0]
        recalled = self._vector_store.search(query_embedding, limit=k_recall)
        candidates = self._dedupe_nodes(recalled)[:k_candidates]
        sections = [
            self._section_source.get_section(hit.record.doc_id, hit.record.node_id)
            for hit in candidates
        ]
        scores_by_node_id = {hit.record.node_id: hit.score for hit in candidates}
        final_sections = self._rank_sections(question=question, sections=sections, k_final=k_final)
        sources = [
            QuerySource(
                doc_id=section.doc_id,
                node_id=section.node_id,
                breadcrumb=section.breadcrumb,
                score=scores_by_node_id.get(section.node_id, 0.0),
                text=section.text,
            )
            for section in final_sections
        ]
        return QueryResponse(
            answer=self._synthesize(question=question, sections=final_sections),
            sources=sources,
        )

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
