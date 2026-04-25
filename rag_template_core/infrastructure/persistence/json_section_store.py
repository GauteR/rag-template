from __future__ import annotations

import json
from pathlib import Path

from rag_template_core.application.ports.section_source import SectionSourcePort
from rag_template_core.domain.models import Document, Section


class JsonSectionStore(SectionSourcePort):
    def __init__(self, *, path: Path) -> None:
        self._path = path
        self._sections: dict[tuple[str, str], Section] = {}
        self._load()

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
        self._save()

    def delete_document(self, doc_id: str) -> None:
        self._sections = {
            key: section for key, section in self._sections.items() if section.doc_id != doc_id
        }
        self._save()

    def get_section(self, doc_id: str, node_id: str) -> Section:
        return self._sections[(doc_id, node_id)]

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw_sections = json.loads(self._path.read_text(encoding="utf-8"))
        self._sections = {
            (item["doc_id"], item["node_id"]): Section(
                doc_id=item["doc_id"],
                node_id=item["node_id"],
                breadcrumb=tuple(item["breadcrumb"]),
                text=item["text"],
            )
            for item in raw_sections
        }

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "doc_id": section.doc_id,
                "node_id": section.node_id,
                "breadcrumb": list(section.breadcrumb),
                "text": section.text,
            }
            for section in self._sections.values()
        ]
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
