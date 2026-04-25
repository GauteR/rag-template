from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuerySource:
    doc_id: str
    node_id: str
    breadcrumb: tuple[str, ...]
    score: float
    text: str


@dataclass(frozen=True)
class QueryResponse:
    answer: str
    sources: list[QuerySource]
