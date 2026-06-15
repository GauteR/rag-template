from __future__ import annotations

import json
from pathlib import Path

from core.application.ports.section_source import SectionSourcePort
from core.domain.models import Document, Section


class JsonSectionStore(SectionSourcePort):
    def __init__(self, *, path: Path) -> None:
        self._path = path
        self._sections: dict[tuple[str, str], Section] = {}
        self._load()

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
            self._save()
        except Exception:
            self._sections = original_sections
            raise

    def delete_document(self, doc_id: str) -> None:
        self._sections = {
            key: section for key, section in self._sections.items() if section.doc_id != doc_id
        }
        self._save()

    def get_section(self, doc_id: str, node_id: str) -> Section:
        return self._sections[(doc_id, node_id)]

    def doc_ids(self) -> set[str]:
        return {doc_id for doc_id, _ in self._sections}

    def section_counts_by_doc(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc_id, _ in self._sections:
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

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
                citation=item.get("citation"),
                start_offset=item.get("start_offset"),
                end_offset=item.get("end_offset"),
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
                "citation": section.citation,
                "start_offset": section.start_offset,
                "end_offset": section.end_offset,
            }
            for section in self._sections.values()
        ]
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
