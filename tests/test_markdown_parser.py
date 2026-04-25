from core.application.indexing.chunking import StructureGuidedChunker
from core.application.indexing.markdown_parser import MarkdownSkeletonParser


def test_markdown_parser_builds_heading_tree_and_breadcrumbs() -> None:
    parser = MarkdownSkeletonParser()

    document = parser.parse(
        doc_id="manual",
        markdown=(
            "# Intro\nOpening\n\n## Install\nRun it\n\n### Windows\nUse PowerShell\n\n## Query\nAsk"
        ),
    )

    intro, install, windows, query = document.nodes

    assert intro.parent_id is None
    assert install.parent_id == intro.node_id
    assert windows.parent_id == install.node_id
    assert query.parent_id == intro.node_id
    assert windows.breadcrumb == ("Intro", "Install", "Windows")
    assert query.content.strip() == "Ask"


def test_markdown_parser_degrades_to_root_node_without_headings() -> None:
    parser = MarkdownSkeletonParser()

    document = parser.parse(doc_id="flat", markdown="Only body text\nwith two lines")

    assert len(document.nodes) == 1
    assert document.nodes[0].node_id == "flat:root"
    assert document.nodes[0].title == "flat"
    assert document.nodes[0].content == "Only body text\nwith two lines"


def test_markdown_parser_keeps_preamble_with_first_section() -> None:
    parser = MarkdownSkeletonParser()

    document = parser.parse(doc_id="manual", markdown="Lead text\n\n# Intro\nOpening")

    assert document.nodes[0].content == "Lead text\n\nOpening"


def test_structure_guided_chunker_injects_breadcrumbs_and_stays_within_node() -> None:
    parser = MarkdownSkeletonParser()
    chunker = StructureGuidedChunker(max_chars=12)
    document = parser.parse(
        doc_id="manual", markdown="# Intro\nAlpha beta gamma\n\n## Install\nDelta"
    )

    chunks = chunker.chunk(document)

    assert {chunk.node_id for chunk in chunks} == {"manual:n1", "manual:n2"}
    assert all(chunk.embedding_text.startswith("Intro") for chunk in chunks)
    assert any("Intro > Install" in chunk.embedding_text for chunk in chunks)
    assert all("Delta" not in chunk.text for chunk in chunks if chunk.node_id == "manual:n1")
