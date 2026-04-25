from __future__ import annotations

from typing import Protocol

from rag_template_core.domain.models import Section


class LlmPort(Protocol):
    def filter_noise(self, *, sections: list[Section]) -> set[str]:
        """Return node IDs that should be excluded from indexing."""

    def rerank(
        self,
        *,
        question: str,
        candidates: list[Section],
        k_final: int,
    ) -> list[str]:
        """Return ranked node IDs for the final pointer context."""

    def synthesize(self, *, question: str, sections: list[Section]) -> str:
        """Return a grounded answer using the provided full sections."""
