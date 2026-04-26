from __future__ import annotations

from core.application.ports.section_source import SectionSourcePort
from core.domain.models import Document, Section


class InMemorySectionStore(SectionSourcePort):
    def __init__(self) -> None:
        self._sections: dict[tuple[str, str], Section] = {}

    def store_document(self, document: Document) -> None:
        nodes = list(document.nodes)
        for index, node in enumerate(nodes):
            descendants = [
                candidate
                for candidate in nodes[index + 1 :]
                if candidate.level > node.level
                and candidate.breadcrumb[: len(node.breadcrumb)] == node.breadcrumb
            ]
            descendant_text = [candidate.section_text for candidate in descendants]
            section_text = "\n\n".join([node.section_text, *descendant_text]).strip()

            end_char = (
                max(d.end_char for d in descendants if d.end_char is not None)
                if descendants
                else node.end_char
            )

            self._sections[(document.doc_id, node.node_id)] = Section(
                doc_id=document.doc_id,
                node_id=node.node_id,
                breadcrumb=node.breadcrumb,
                text=section_text,
                citation=node.citation,
                start_offset=node.start_char,
                end_offset=end_char,
            )

    def delete_document(self, doc_id: str) -> None:
        self._sections = {
            key: section for key, section in self._sections.items() if section.doc_id != doc_id
        }

    def get_section(self, doc_id: str, node_id: str) -> Section:
        return self._sections[(doc_id, node_id)]
