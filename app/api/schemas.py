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


class QuerySourceResponse(BaseModel):
    doc_id: str
    node_id: str
    breadcrumb: list[str]
    score: float
    text: str
    citation: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class QueryResponseModel(BaseModel):
    answer: str
    sources: list[QuerySourceResponse]


class HealthResponse(BaseModel):
    status: str
    routing_provider: str
    llm_provider: str
    embedding_provider: str
    index_document_count: int
    index_ready: bool
    index_consistent: bool
    config_errors: list[str] = Field(default_factory=list)
