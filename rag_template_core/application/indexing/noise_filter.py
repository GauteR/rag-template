from __future__ import annotations

from typing import Protocol

from rag_template_core.application.ports.llm import LlmPort
from rag_template_core.domain.models import Document, Section


class NoiseFilter(Protocol):
    def filter(self, document: Document) -> Document:
        """Return a document with noisy nodes removed."""


class HeuristicNoiseFilter(NoiseFilter):
    def __init__(self, *, noisy_titles: set[str] | None = None) -> None:
        self._noisy_titles = noisy_titles or {
            "appendix",
            "disclaimer",
            "legal",
            "references",
        }

    def filter(self, document: Document) -> Document:
        filtered_nodes = [
            node for node in document.nodes if node.title.strip().lower() not in self._noisy_titles
        ]
        return Document(doc_id=document.doc_id, nodes=tuple(filtered_nodes))


class LlmNoiseFilter(NoiseFilter):
    def __init__(self, *, llm: LlmPort) -> None:
        self._llm = llm

    def filter(self, document: Document) -> Document:
        sections = [
            Section(
                doc_id=node.doc_id,
                node_id=node.node_id,
                breadcrumb=node.breadcrumb,
                text=node.section_text,
            )
            for node in document.nodes
        ]
        noisy_node_ids = self._llm.filter_noise(sections=sections)
        return Document(
            doc_id=document.doc_id,
            nodes=tuple(node for node in document.nodes if node.node_id not in noisy_node_ids),
        )
