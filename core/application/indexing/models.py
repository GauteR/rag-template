from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexMarkdownResult:
    doc_id: str
    indexed_chunks: int
    indexed_sections: int
