from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)

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
    config_errors = container.collect_config_errors()
    index_document_count = 0
    index_consistent = False
    index_ready = False

    try:
        vector_doc_ids = container.vector_store.doc_ids()
        section_doc_ids = container.section_store.doc_ids()
        index_document_count = len(vector_doc_ids)
        index_consistent = vector_doc_ids == section_doc_ids
        index_ready = index_document_count > 0 and index_consistent
    except Exception as exc:
        config_errors.append(f"Index inspection failed: {exc}")

    return HealthResponse(
        status="ok" if not config_errors else "degraded",
        routing_provider=container.settings.routing_provider,
        llm_provider=container.settings.synthesis_provider,
        embedding_provider=container.settings.embedding_provider,
        index_document_count=index_document_count,
        index_ready=index_ready,
        index_consistent=index_consistent,
        config_errors=config_errors,
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
                citation=source.citation,
                start_offset=source.start_offset,
                end_offset=source.end_offset,
            )
            for source in result.sources
        ],
    )


@router.post("/index/pdf", response_model=IndexMarkdownResponse)
async def index_pdf(
    file: Annotated[UploadFile, File(...)],
    doc_id: Annotated[str | None, Form()] = None,
    container: AppContainer = ContainerDependency,
) -> IndexMarkdownResponse:
    if not container.settings.enable_llamaparse:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF indexing is disabled. Set ENABLE_LLAMAPARSE=true to enable it.",
        )

    filename = file.filename or "document.pdf"
    resolved_doc_id = (doc_id or Path(filename).stem).strip()
    if not resolved_doc_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="doc_id is required when the uploaded file has no filename.",
        )

    max_bytes = container.settings.max_upload_mb * 1024 * 1024
    chunk_size = 1024 * 1024
    total_bytes = 0
    content_buffer = bytearray()

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"PDF payload exceeds maximum size of {container.settings.max_upload_mb} MB",
            )
        content_buffer.extend(chunk)

    if not content_buffer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF payload is empty.",
        )

    content = bytes(content_buffer)

    try:
        markdown = container.pdf_extractor.extract_markdown(filename=filename, content=content)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    result = container.index_markdown_use_case().execute(
        doc_id=resolved_doc_id,
        markdown=markdown,
    )
    return IndexMarkdownResponse(
        doc_id=result.doc_id,
        indexed_chunks=result.indexed_chunks,
        indexed_sections=result.indexed_sections,
    )
