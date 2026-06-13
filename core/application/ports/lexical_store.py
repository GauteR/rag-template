from __future__ import annotations

from typing import Protocol

from core.domain.models import SearchHit


class LexicalStorePort(Protocol):
    def index_document(self, *, doc_id: str, records: list[tuple[str, str, str]]) -> None:
        """Index chunk text keyed by doc_id, node_id, chunk_id."""

    def delete_document(self, doc_id: str) -> None:
        """Remove all lexical records for a document."""

    def search(self, *, query: str, limit: int, doc_id: str | None = None) -> list[SearchHit]:
        """Return lexical matches as SearchHit objects."""
