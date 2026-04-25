from __future__ import annotations

from core.application.ports.vector_store import VectorStorePort
from core.domain.models import SearchHit, VectorRecord
from core.infrastructure.persistence.in_memory_vector_store import InMemoryVectorStore


def normalize_vector(vector: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


class FaissVectorStore(VectorStorePort):
    def __init__(self, *, dimension: int) -> None:
        self._dimension = dimension
        self._records: list[VectorRecord] = []
        self._fallback = InMemoryVectorStore()
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError:
            self._faiss = None
            self._index = None
        else:
            self._faiss = faiss
            self._index = faiss.IndexFlatIP(dimension)

    def add(self, records: list[VectorRecord]) -> None:
        for record in records:
            if len(record.embedding) != self._dimension:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {self._dimension}, got {len(record.embedding)}"
                )
        if self._index is None:
            self._fallback.add(records)
            return
        import numpy as np

        normalized_records = [
            VectorRecord(
                doc_id=record.doc_id,
                node_id=record.node_id,
                chunk_id=record.chunk_id,
                embedding=normalize_vector(record.embedding),
                text=record.text,
                breadcrumb=record.breadcrumb,
            )
            for record in records
        ]
        vectors = np.array([record.embedding for record in normalized_records], dtype="float32")
        self._index.add(vectors)
        self._records.extend(normalized_records)

    def delete_document(self, doc_id: str) -> None:
        if self._index is None:
            self._fallback.delete_document(doc_id)
            return
        self._records = [record for record in self._records if record.doc_id != doc_id]
        self._rebuild_index()

    def search(self, embedding: list[float], *, limit: int) -> list[SearchHit]:
        if self._index is None:
            return self._fallback.search(embedding, limit=limit)
        import numpy as np

        query = np.array([normalize_vector(embedding)], dtype="float32")
        scores, indices = self._index.search(query, limit)
        return [
            SearchHit(record=self._records[index], score=float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]

    @property
    def dimension(self) -> int:
        return self._dimension

    def count(self) -> int:
        if self._index is None:
            return self._fallback.count()
        return len(self._records)

    def _rebuild_index(self) -> None:
        if self._faiss is None:
            return
        self._index = self._faiss.IndexFlatIP(self._dimension)
        if not self._records:
            return
        import numpy as np

        vectors = np.array([record.embedding for record in self._records], dtype="float32")
        self._index.add(vectors)
