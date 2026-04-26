from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentNode:
    doc_id: str
    node_id: str
    title: str
    level: int
    order: int
    content: str
    parent_id: str | None
    breadcrumb: tuple[str, ...]
    start_char: int | None = None
    end_char: int | None = None

    @property
    def section_text(self) -> str:
        if self.level <= 0:
            return self.content
        heading = f"{'#' * self.level} {self.title}"
        return f"{heading}\n{self.content}".strip()


@dataclass(frozen=True)
class Document:
    doc_id: str
    nodes: tuple[DocumentNode, ...]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    node_id: str
    chunk_id: str
    text: str
    embedding_text: str
    breadcrumb: tuple[str, ...]


@dataclass(frozen=True)
class Section:
    doc_id: str
    node_id: str
    breadcrumb: tuple[str, ...]
    text: str
    citation: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass(frozen=True)
class VectorRecord:
    doc_id: str
    node_id: str
    chunk_id: str
    embedding: tuple[float, ...]
    text: str
    breadcrumb: tuple[str, ...]


@dataclass(frozen=True)
class SearchHit:
    record: VectorRecord
    score: float
