from __future__ import annotations

from core.domain.models import Chunk, Document


class StructureGuidedChunker:
    def __init__(self, *, max_chars: int = 1_200) -> None:
        self._max_chars = max_chars

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for node in document.nodes:
            for index, text in enumerate(self._split_text(node.content), start=1):
                breadcrumb_text = " > ".join(node.breadcrumb)
                chunks.append(
                    Chunk(
                        doc_id=document.doc_id,
                        node_id=node.node_id,
                        chunk_id=f"{node.node_id}:c{index}",
                        text=text,
                        embedding_text=f"{breadcrumb_text}\n{text}".strip(),
                        breadcrumb=node.breadcrumb,
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        clean_text = text.strip()
        if not clean_text:
            return [""]
        if len(clean_text) <= self._max_chars:
            return [clean_text]

        words = clean_text.split()
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0
        for word in words:
            projected_length = current_length + len(word) + (1 if current else 0)
            if current and projected_length > self._max_chars:
                chunks.append(" ".join(current))
                current = [word]
                current_length = len(word)
            else:
                current.append(word)
                current_length = projected_length

        if current:
            chunks.append(" ".join(current))
        return chunks
