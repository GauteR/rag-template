from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field


class McpServerAdapter:
    name: str
    tool_names: set[str]

    def tool(self, *, name: str, annotations: dict[str, object] | None = None):
        raise NotImplementedError

    def run(self, *, transport: str) -> None:
        raise NotImplementedError


@dataclass
class TestableMcpServer(McpServerAdapter):
    name: str
    tool_names: set[str] = field(default_factory=set)
    tools: dict[str, Any] = field(default_factory=dict)

    def tool(self, *, name: str, annotations: dict[str, object] | None = None):
        del annotations

        def decorator(func):
            self.tool_names.add(name)
            self.tools[name] = func
            return func

        return decorator

    def run(self, *, transport: str) -> None:
        raise RuntimeError(
            f"FastMCP is required to run transport '{transport}'. "
            "Install with `uv sync --extra mcp`."
        )


class FastMcpServerAdapter(McpServerAdapter):
    def __init__(self, *, name: str) -> None:
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            raise RuntimeError("Install the mcp extra with `uv sync --extra mcp`.") from exc

        self.name = name
        self.tool_names: set[str] = set()
        self._server = FastMCP(name)

    def tool(self, *, name: str, annotations: dict[str, object] | None = None):
        self.tool_names.add(name)
        return self._server.tool(name=name, annotations=annotations)

    def run(self, *, transport: str) -> None:
        self._server.run(transport=transport)


class HealthInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexMarkdownInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    doc_id: str = Field(..., min_length=1, description="Document identifier to index.")
    markdown: str = Field(..., min_length=1, description="Markdown content to index.")


class IndexPdfInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    filename: str = Field(..., min_length=1, description="PDF filename, e.g. manual.pdf.")
    content_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded PDF bytes.",
    )
    doc_id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional document identifier. Defaults to filename stem.",
    )


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(..., min_length=1, description="Question to answer from indexed content.")
    k_recall: int = Field(default=200, ge=1, le=1000)
    k_candidates: int = Field(default=50, ge=1, le=500)
    k_final: int = Field(default=5, ge=1, le=50)


class RagApiClient:
    def __init__(self, *, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key} if api_key else {}

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/health")

    async def index_markdown(self, *, doc_id: str, markdown: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/index/markdown",
            json={"doc_id": doc_id, "markdown": markdown},
        )

    async def index_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        doc_id: str | None,
    ) -> dict[str, Any]:
        data: dict[str, str] = {}
        if doc_id is not None:
            data["doc_id"] = doc_id
        files = {"file": (filename, content, "application/pdf")}
        return await self._request("POST", "/v1/index/pdf", data=data, files=files)

    async def query(
        self,
        *,
        question: str,
        k_recall: int,
        k_candidates: int,
        k_final: int,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/query",
            json={
                "question": question,
                "k_recall": k_recall,
                "k_candidates": k_candidates,
                "k_final": k_final,
            },
        )

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(base_url=self._base_url, headers=self._headers) as client:
                response = await client.request(method, path, timeout=30, **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"RAG API returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"RAG API request failed: {exc}") from exc


def create_mcp_server(
    *,
    base_url: str,
    api_key: str | None = None,
    runtime: bool = False,
) -> McpServerAdapter:
    server: McpServerAdapter
    server = (
        FastMcpServerAdapter(name="rag_template_mcp")
        if runtime
        else TestableMcpServer(name="rag_template_mcp")
    )
    client = RagApiClient(base_url=base_url, api_key=api_key)

    @server.tool(
        name="rag_health",
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    )
    async def rag_health(params: HealthInput) -> dict[str, Any]:
        """Check RAG API health and active provider configuration."""
        del params
        return await client.health()

    @server.tool(
        name="rag_index_markdown",
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    )
    async def rag_index_markdown(params: IndexMarkdownInput) -> dict[str, Any]:
        """Index Markdown content in the Proxy-Pointer RAG service."""
        return await client.index_markdown(doc_id=params.doc_id, markdown=params.markdown)

    @server.tool(
        name="rag_index_pdf",
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    )
    async def rag_index_pdf(params: IndexPdfInput) -> dict[str, Any]:
        """Index PDF bytes as Markdown in the standard indexing pipeline."""
        try:
            content = base64.b64decode(params.content_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError(
                "Failed to decode content_base64: invalid base64 format."
            ) from exc
        return await client.index_pdf(
            filename=params.filename,
            content=content,
            doc_id=params.doc_id,
        )

    @server.tool(
        name="rag_query",
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    )
    async def rag_query(params: QueryInput) -> dict[str, Any]:
        """Ask a grounded question and return answer plus traceable sources."""
        return await client.query(
            question=params.question,
            k_recall=params.k_recall,
            k_candidates=params.k_candidates,
            k_final=params.k_final,
        )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose rag-template as an MCP server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    args = parser.parse_args()

    create_mcp_server(base_url=args.base_url, api_key=args.api_key, runtime=True).run(
        transport=args.transport
    )


if __name__ == "__main__":
    main()
