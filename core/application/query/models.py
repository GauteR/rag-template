from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuerySource:
    doc_id: str
    node_id: str
    breadcrumb: tuple[str, ...]
    score: float
    text: str
    citation: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass
class QueryMetadata:
    recall_ms: float = 0.0
    rerank_ms: float = 0.0
    synthesize_ms: float = 0.0
    recalled_count: int = 0
    candidate_count: int = 0
    final_count: int = 0
    enable_llm_reranker: bool = False
    enable_hybrid_search: bool = False


@dataclass(frozen=True)
class QueryResponse:
    answer: str
    sources: list[QuerySource]
    metadata: QueryMetadata | None = None
