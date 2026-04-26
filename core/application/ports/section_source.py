from __future__ import annotations

from typing import Protocol

from core.domain.models import Document, Section


class SectionSourcePort(Protocol):
    def store_document(self, document: Document) -> None:
        """Persist all node sections for later pointer fetch."""

    def delete_document(self, doc_id: str) -> None:
        """Remove persisted sections for a document before replacement indexing."""

    def get_section(self, doc_id: str, node_id: str) -> Section:
        """Fetch a full section by document and node ID."""

    def doc_ids(self) -> set[str]:
        """Return the set of document IDs that have persisted sections."""
