from __future__ import annotations

from pydantic import BaseModel, Field


class IndexMarkdownRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    markdown: str = Field(min_length=1)


class IndexMarkdownResponse(BaseModel):
    doc_id: str
    indexed_chunks: int
    indexed_sections: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k_recall: int = Field(default=200, ge=1, le=1_000)
    k_candidates: int = Field(default=50, ge=1, le=500)
    k_final: int = Field(default=5, ge=1, le=50)
    doc_id: str | None = None
    min_score: float | None = Field(default=None, ge=0.0)


class QuerySourceResponse(BaseModel):
    doc_id: str
    node_id: str
    breadcrumb: list[str]
    score: float
    text: str
    citation: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class QueryMetadataResponse(BaseModel):
    recall_ms: float = 0.0
    rerank_ms: float = 0.0
    synthesize_ms: float = 0.0
    recalled_count: int = 0
    candidate_count: int = 0
    final_count: int = 0
    enable_llm_reranker: bool = False
    enable_hybrid_search: bool = False


class QueryResponseModel(BaseModel):
    answer: str
    sources: list[QuerySourceResponse]
    metadata: QueryMetadataResponse | None = None


class IndexDocumentSummary(BaseModel):
    doc_id: str
    section_count: int
    chunk_count: int


class IndexListResponse(BaseModel):
    documents: list[IndexDocumentSummary]


class DeleteIndexResponse(BaseModel):
    doc_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    routing_provider: str
    llm_provider: str
    embedding_provider: str
    vector_store_provider: str
    faiss_available: bool
    chroma_reachable: bool | None = None
    index_document_count: int
    index_ready: bool
    index_consistent: bool
    config_errors: list[str] = Field(default_factory=list)
