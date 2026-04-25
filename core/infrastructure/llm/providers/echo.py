from __future__ import annotations

from core.application.ports.llm import LlmPort
from core.domain.models import Section


class EchoLlm(LlmPort):
    def filter_noise(self, *, sections: list[Section]) -> set[str]:
        del sections
        return set()

    def rerank(self, *, question: str, candidates: list[Section], k_final: int) -> list[str]:
        del question
        return [section.node_id for section in candidates[:k_final]]

    def synthesize(self, *, question: str, sections: list[Section]) -> str:
        if not sections:
            return "No relevant sections found."
        context = "\n\n".join(section.text for section in sections)
        return f"Question: {question}\n\nContext:\n{context}"
