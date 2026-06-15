from __future__ import annotations

from typing import Protocol

from core.domain.models import SearchHit, VectorRecord


class VectorStorePort(Protocol):
    def add(self, records: list[VectorRecord]) -> None:
        """Add embedded chunks to the vector index."""

    def replace_document(self, doc_id: str, records: list[VectorRecord]) -> None:
        """Atomically replace all vectors for a document."""

    def delete_document(self, doc_id: str) -> None:
        """Remove all vectors for a document before replacement indexing."""

    def search(self, embedding: list[float], *, limit: int) -> list[SearchHit]:
        """Return the nearest records for the query vector."""

    def count(self) -> int:
        """Return the total number of stored vectors."""

    def doc_ids(self) -> set[str]:
        """Return the set of document IDs present in the vector index."""

    def chunk_counts_by_doc(self) -> dict[str, int]:
        """Return chunk counts grouped by document ID."""
