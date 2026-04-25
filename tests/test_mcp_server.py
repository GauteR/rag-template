from core.mcp_server import create_mcp_server


def test_create_mcp_server_registers_rag_tools() -> None:
    server = create_mcp_server(base_url="http://rag.local", api_key="secret")

    assert server.name == "rag_template_mcp"
    assert {"rag_health", "rag_index_markdown", "rag_query"}.issubset(server.tool_names)
