from __future__ import annotations

from core.application.ports.section_source import SectionSourcePort
from core.domain.models import Document, Section


class InMemorySectionStore(SectionSourcePort):
    def __init__(self) -> None:
        self._sections: dict[tuple[str, str], Section] = {}

    def store_document(self, document: Document) -> None:
        self.replace_document(document)

    def replace_document(self, document: Document) -> None:
        original_sections = dict(self._sections)
        try:
            new_sections = self._build_sections(document)
            self._sections = {
                key: section for key, section in self._sections.items() if key[0] != document.doc_id
            }
            self._sections.update(new_sections)
        except Exception:
            self._sections = original_sections
            raise

    def delete_document(self, doc_id: str) -> None:
        self._sections = {
            key: section for key, section in self._sections.items() if section.doc_id != doc_id
        }

    def get_section(self, doc_id: str, node_id: str) -> Section:
        return self._sections[(doc_id, node_id)]

    def doc_ids(self) -> set[str]:
        return {doc_id for doc_id, _ in self._sections}

    def section_counts_by_doc(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc_id, _ in self._sections:
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    def _build_sections(self, document: Document) -> dict[tuple[str, str], Section]:
        sections: dict[tuple[str, str], Section] = {}
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

            sections[(document.doc_id, node.node_id)] = Section(
                doc_id=document.doc_id,
                node_id=node.node_id,
                breadcrumb=node.breadcrumb,
                text=section_text,
                citation=node.citation,
                start_offset=node.start_char,
                end_offset=end_char,
            )
        return sections
