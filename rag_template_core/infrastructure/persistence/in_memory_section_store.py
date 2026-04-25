from __future__ import annotations

from rag_template_core.application.ports.section_source import SectionSourcePort
from rag_template_core.domain.models import Document, Section


class InMemorySectionStore(SectionSourcePort):
    def __init__(self) -> None:
        self._sections: dict[tuple[str, str], Section] = {}

    def store_document(self, document: Document) -> None:
        nodes = list(document.nodes)
        for index, node in enumerate(nodes):
            descendant_text = [
                candidate.section_text
                for candidate in nodes[index + 1 :]
                if candidate.level > node.level
                and candidate.breadcrumb[: len(node.breadcrumb)] == node.breadcrumb
            ]
            section_text = "\n\n".join([node.section_text, *descendant_text]).strip()
            self._sections[(document.doc_id, node.node_id)] = Section(
                doc_id=document.doc_id,
                node_id=node.node_id,
                breadcrumb=node.breadcrumb,
                text=section_text,
            )

    def delete_document(self, doc_id: str) -> None:
        self._sections = {
            key: section for key, section in self._sections.items() if section.doc_id != doc_id
        }

    def get_section(self, doc_id: str, node_id: str) -> Section:
        return self._sections[(doc_id, node_id)]
