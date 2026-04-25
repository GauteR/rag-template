from __future__ import annotations

from typing import Protocol

from rag_template_core.domain.models import SearchHit, VectorRecord


class VectorStorePort(Protocol):
    def add(self, records: list[VectorRecord]) -> None:
        """Add embedded chunks to the vector index."""

    def delete_document(self, doc_id: str) -> None:
        """Remove all vectors for a document before replacement indexing."""

    def search(self, embedding: list[float], *, limit: int) -> list[SearchHit]:
        """Return the nearest records for the query vector."""
