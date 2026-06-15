from __future__ import annotations

import json
import logging
import re
import secrets
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
from fastapi import (
    Path as PathParam,
)
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    DOC_ID_PATTERN,
    DeleteIndexResponse,
    HealthResponse,
    IndexDocumentSummary,
    IndexListResponse,
    IndexMarkdownRequest,
    IndexMarkdownResponse,
    QueryMetadataResponse,
    QueryRequest,
    QueryResponseModel,
    QuerySourceResponse,
)
from app.container import AppContainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_authorized_container(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> AppContainer:
    container = get_container(request)
    if container.settings.api_key and (
        x_api_key is None or not secrets.compare_digest(x_api_key, container.settings.api_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
        )
    return container


ContainerDependency = Depends(get_authorized_container)


def _validate_doc_id(doc_id: str) -> str:
    if not re.fullmatch(DOC_ID_PATTERN, doc_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="doc_id contains invalid characters",
        )
    return doc_id


def _validate_question_length(*, question: str, max_chars: int) -> None:
    if len(question) > max_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"question exceeds maximum length of {max_chars} characters",
        )


def _require_index_admin(container: AppContainer) -> None:
    if not container.settings.enable_index_admin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Index admin is disabled. Set ENABLE_INDEX_ADMIN=true to enable it.",
        )


def _to_query_source_responses(sources) -> list[QuerySourceResponse]:
    return [
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
        for source in sources
    ]


def _to_query_response(result) -> QueryResponseModel:
    metadata = None
    if result.metadata is not None:
        metadata = QueryMetadataResponse(
            recall_ms=result.metadata.recall_ms,
            rerank_ms=result.metadata.rerank_ms,
            synthesize_ms=result.metadata.synthesize_ms,
            recalled_count=result.metadata.recalled_count,
            candidate_count=result.metadata.candidate_count,
            final_count=result.metadata.final_count,
            enable_llm_reranker=result.metadata.enable_llm_reranker,
            enable_hybrid_search=result.metadata.enable_hybrid_search,
        )
    return QueryResponseModel(
        answer=result.answer,
        sources=_to_query_source_responses(result.sources),
        metadata=metadata,
    )


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> HealthResponse:
    container = get_container(request)
    if container.settings.api_key and not container.settings.enable_public_health:
        if x_api_key is None or not secrets.compare_digest(x_api_key, container.settings.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key",
            )
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
        vector_store_provider=container.settings.vector_store_provider,
        faiss_available=container.faiss_available(),
        chroma_reachable=container.chroma_reachable(),
        index_document_count=index_document_count,
        index_ready=index_ready,
        index_consistent=index_consistent,
        config_errors=config_errors,
    )


@router.get("/index", response_model=IndexListResponse)
def list_index(container: AppContainer = ContainerDependency) -> IndexListResponse:
    _require_index_admin(container)
    section_counts = container.section_store.section_counts_by_doc()
    chunk_counts = container.vector_store.chunk_counts_by_doc()
    doc_ids = sorted(section_counts.keys() | chunk_counts.keys())
    return IndexListResponse(
        documents=[
            IndexDocumentSummary(
                doc_id=doc_id,
                section_count=section_counts.get(doc_id, 0),
                chunk_count=chunk_counts.get(doc_id, 0),
            )
            for doc_id in doc_ids
        ]
    )


@router.delete("/index/{doc_id}", response_model=DeleteIndexResponse)
def delete_index(
    doc_id: Annotated[str, PathParam(pattern=DOC_ID_PATTERN)],
    container: AppContainer = ContainerDependency,
) -> DeleteIndexResponse:
    _require_index_admin(container)
    existed = container.delete_document(doc_id)
    return DeleteIndexResponse(doc_id=doc_id, deleted=existed)


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
    result = container.index_markdown_use_case.execute(
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
    _validate_question_length(
        question=request.question,
        max_chars=container.settings.max_query_chars,
    )
    result = container.query_use_case.execute(
        question=request.question,
        k_recall=request.k_recall,
        k_candidates=request.k_candidates,
        k_final=request.k_final,
        doc_id=request.doc_id,
        min_score=request.min_score,
    )
    if result.metadata is not None:
        logger.info(
            "query completed",
            extra={
                "recall_ms": result.metadata.recall_ms,
                "rerank_ms": result.metadata.rerank_ms,
                "synthesize_ms": result.metadata.synthesize_ms,
                "recalled_count": result.metadata.recalled_count,
                "candidate_count": result.metadata.candidate_count,
                "final_count": result.metadata.final_count,
            },
        )
    return _to_query_response(result)


@router.post("/query/stream")
def query_stream(
    request: QueryRequest,
    container: AppContainer = ContainerDependency,
) -> StreamingResponse:
    if not container.settings.enable_streaming_query:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Streaming query is disabled. Set ENABLE_STREAMING_QUERY=true to enable it.",
        )
    _validate_question_length(
        question=request.question,
        max_chars=container.settings.max_query_chars,
    )

    sources, token_stream = container.query_use_case.synthesize_stream(
        question=request.question,
        k_recall=request.k_recall,
        k_candidates=request.k_candidates,
        k_final=request.k_final,
        doc_id=request.doc_id,
        min_score=request.min_score,
    )

    def event_generator():
        sources_payload = [source.model_dump() for source in _to_query_source_responses(sources)]
        yield f"event: sources\ndata: {json.dumps({'sources': sources_payload})}\n\n"
        for token in token_stream:
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/index/pdf", response_model=IndexMarkdownResponse)
async def index_pdf(
    file: Annotated[UploadFile, File(...)],
    doc_id: Annotated[str | None, Form()] = None,
    container: AppContainer = ContainerDependency,
) -> IndexMarkdownResponse:
    if not container.settings.enable_pdf_indexing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF indexing is disabled. Set ENABLE_PDF_INDEXING=true to enable it.",
        )

    filename = file.filename or "document.pdf"
    resolved_doc_id = _validate_doc_id((doc_id or Path(filename).stem).strip())
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

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file: missing %PDF- header",
        )

    try:
        markdown = container.pdf_extractor.extract_markdown(filename=filename, content=content)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    result = container.index_markdown_use_case.execute(
        doc_id=resolved_doc_id,
        markdown=markdown,
    )
    return IndexMarkdownResponse(
        doc_id=result.doc_id,
        indexed_chunks=result.indexed_chunks,
        indexed_sections=result.indexed_sections,
    )
