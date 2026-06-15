"""Seed the benchmark index from fixtures under benchmarks/fixtures/docs/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.container import AppContainer
from core.application.indexing.markdown_parser import MarkdownSkeletonParser
from core.config.settings import Settings


def seed_index(*, index_dir: Path, fixtures_dir: Path) -> dict[str, list[str]]:
    container = AppContainer(settings=Settings(index_dir=index_dir))
    parser = MarkdownSkeletonParser()
    use_case = container.index_markdown_use_case
    node_ids_by_doc: dict[str, list[str]] = {}

    for markdown_path in sorted(fixtures_dir.glob("*.md")):
        doc_id = markdown_path.stem
        markdown = markdown_path.read_text(encoding="utf-8")
        use_case.execute(doc_id=doc_id, markdown=markdown)
        document = parser.parse(doc_id=doc_id, markdown=markdown)
        node_ids_by_doc[doc_id] = [node.node_id for node in document.nodes]

    return node_ids_by_doc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed benchmark fixtures into the index.")
    parser.add_argument(
        "--index-dir",
        default="benchmarks/.benchmark-index",
        help="Directory for the seeded benchmark index",
    )
    parser.add_argument(
        "--fixtures-dir",
        default="benchmarks/fixtures/docs",
        help="Directory containing fixture markdown files",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).parent.parent
    index_dir = Path(args.index_dir)
    if not index_dir.is_absolute():
        index_dir = repo_root / index_dir
    fixtures_dir = Path(args.fixtures_dir)
    if not fixtures_dir.is_absolute():
        fixtures_dir = repo_root / fixtures_dir

    node_ids_by_doc = seed_index(index_dir=index_dir, fixtures_dir=fixtures_dir)
    print(f"Seeded index at {index_dir}")
    for doc_id, node_ids in node_ids_by_doc.items():
        print(f"  {doc_id}: {', '.join(node_ids)}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
