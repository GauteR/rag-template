import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from core.mcp_server import create_mcp_server, verify_api_health


def test_create_mcp_server_registers_rag_tools() -> None:
    server = create_mcp_server(base_url="http://rag.local", api_key="secret")

    assert server.name == "rag_template_mcp"
    expected_tools = {
        "rag_health",
        "rag_index_markdown",
        "rag_index_pdf",
        "rag_query",
        "rag_delete_index",
    }
    assert expected_tools.issubset(server.tool_names)


def test_verify_api_health_passes_when_status_ok() -> None:
    mock_client = AsyncMock()
    mock_client.health.return_value = {"status": "ok", "config_errors": []}

    with patch("core.mcp_server.RagApiClient", return_value=mock_client):
        asyncio.run(verify_api_health(base_url="http://rag.local"))

    mock_client.health.assert_awaited_once()


def test_verify_api_health_raises_when_status_degraded() -> None:
    mock_client = AsyncMock()
    mock_client.health.return_value = {
        "status": "degraded",
        "config_errors": ["EMBEDDING_DIMENSION mismatch"],
    }

    with patch("core.mcp_server.RagApiClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="EMBEDDING_DIMENSION mismatch"):
            asyncio.run(verify_api_health(base_url="http://rag.local"))
