from __future__ import annotations

import math

from rag_template_core.application.ports.vector_store import VectorStorePort
from rag_template_core.domain.models import SearchHit, VectorRecord


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(self, records: list[VectorRecord]) -> None:
        self._records.extend(records)

    def delete_document(self, doc_id: str) -> None:
        self._records = [record for record in self._records if record.doc_id != doc_id]

    def search(self, embedding: list[float], *, limit: int) -> list[SearchHit]:
        return [
            SearchHit(record=record, score=score)
            for score, record in sorted(
                (
                    (self._cosine_similarity(embedding, list(record.embedding)), record)
                    for record in self._records
                ),
                key=lambda item: item[0],
                reverse=True,
            )[:limit]
        ]

    def count(self) -> int:
        return len(self._records)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
