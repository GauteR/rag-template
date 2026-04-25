from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from app.api.schemas import (
    HealthResponse,
    IndexMarkdownRequest,
    IndexMarkdownResponse,
    QueryRequest,
    QueryResponseModel,
    QuerySourceResponse,
)
from app.container import AppContainer

router = APIRouter(prefix="/v1")


def get_authorized_container(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> AppContainer:
    container: AppContainer = request.app.state.container
    if container.settings.api_key and x_api_key != container.settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
        )
    return container


ContainerDependency = Depends(get_authorized_container)


@router.get("/health", response_model=HealthResponse)
def health(container: AppContainer = ContainerDependency) -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=container.settings.synthesis_provider,
        embedding_provider=container.settings.embedding_provider,
        index_ready=container.vector_store.count() > 0,
    )


@router.post("/index/markdown", response_model=IndexMarkdownResponse)
def index_markdown(
    request: IndexMarkdownRequest,
    container: AppContainer = ContainerDependency,
) -> IndexMarkdownResponse:
    max_bytes = container.settings.max_upload_mb * 1024 * 1024
    if len(request.markdown.encode("utf-8")) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Markdown payload exceeds MAX_UPLOAD_MB",
        )
    result = container.index_markdown_use_case().execute(
        doc_id=request.doc_id,
        markdown=request.markdown,
    )
    return IndexMarkdownResponse(
        doc_id=result.doc_id,
        indexed_chunks=result.indexed_chunks,
        indexed_sections=result.indexed_sections,
    )


@router.post("/query", response_model=QueryResponseModel)
def query(
    request: QueryRequest,
    container: AppContainer = ContainerDependency,
) -> QueryResponseModel:
    result = container.query_use_case().execute(
        question=request.question,
        k_recall=request.k_recall,
        k_candidates=request.k_candidates,
        k_final=request.k_final,
    )
    return QueryResponseModel(
        answer=result.answer,
        sources=[
            QuerySourceResponse(
                doc_id=source.doc_id,
                node_id=source.node_id,
                breadcrumb=list(source.breadcrumb),
                score=source.score,
                text=source.text,
            )
            for source in result.sources
        ],
    )


@router.post("/index/pdf")
def index_pdf(container: AppContainer = ContainerDependency) -> dict[str, str]:
    if not container.settings.enable_llamaparse:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF indexing is disabled. Set ENABLE_LLAMAPARSE=true to enable it.",
        )
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="PDF indexing requires a configured PdfExtractorPort adapter.",
    )
