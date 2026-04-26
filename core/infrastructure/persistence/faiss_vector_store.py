from __future__ import annotations

import json
from pathlib import Path

from core.application.ports.vector_store import VectorStorePort
from core.domain.models import SearchHit, VectorRecord
from core.infrastructure.persistence.in_memory_vector_store import InMemoryVectorStore


def normalize_vector(vector: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


class FaissVectorStore(VectorStorePort):
    def __init__(self, *, dimension: int, index_path: Path | None = None) -> None:
        self._dimension = dimension
        self._records: list[VectorRecord] = []
        self._fallback = InMemoryVectorStore()
        self._index_path = index_path
        self._records_path = None
        if index_path is not None:
            self._records_path = index_path.with_name(f"{index_path.stem}.records.json")
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError:
            self._faiss = None
            self._index = None
        else:
            self._faiss = faiss
            self._index = faiss.IndexFlatIP(dimension)
        self._load()

    def add(self, records: list[VectorRecord]) -> None:
        for record in records:
            if len(record.embedding) != self._dimension:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {self._dimension}, got {len(record.embedding)}"
                )
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
        if self._index is None:
            self._fallback.add(normalized_records)
        else:
            import numpy as np

            vectors = np.array([record.embedding for record in normalized_records], dtype="float32")
            self._index.add(vectors)
        self._records.extend(normalized_records)
        self._save()

    def delete_document(self, doc_id: str) -> None:
        if self._index is None:
            self._records = [record for record in self._records if record.doc_id != doc_id]
            self._fallback.delete_document(doc_id)
        else:
            original_records = self._records
            original_index = self._index
            self._records = [record for record in self._records if record.doc_id != doc_id]
            try:
                self._rebuild_index()
            except Exception:
                self._records = original_records
                self._index = original_index
                raise
        self._save()

    def search(self, embedding: list[float], *, limit: int) -> list[SearchHit]:
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {len(embedding)}"
            )
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

    def _load(self) -> None:
        if self._records_path is None:
            return
        index_exists = self._index_path is not None and self._index_path.exists()
        records_exists = self._records_path.exists()
        if not index_exists and not records_exists:
            return
        if not records_exists:
            raise ValueError(
                "Inconsistent persisted vector index state: "
                "records file is missing while index file exists."
            )
        try:
            payload = json.loads(self._records_path.read_text(encoding="utf-8"))
            self._records = [
                VectorRecord(
                    doc_id=item["doc_id"],
                    node_id=item["node_id"],
                    chunk_id=item["chunk_id"],
                    embedding=tuple(item["embedding"]),
                    text=item["text"],
                    breadcrumb=tuple(item["breadcrumb"]),
                )
                for item in payload
            ]
        except (TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Failed to load persisted vector records from {self._records_path}"
            ) from exc
        for record in self._records:
            if len(record.embedding) != self._dimension:
                raise ValueError(
                    "Embedding dimension mismatch in persisted records: "
                    f"expected {self._dimension}, got {len(record.embedding)}"
                )
        if self._index is None:
            self._fallback.add(self._records)
            return
        if index_exists:
            try:
                self._index = self._faiss.read_index(str(self._index_path))
            except (RuntimeError, OSError, ValueError) as exc:
                raise ValueError(f"Failed to load FAISS index from {self._index_path}") from exc
        else:
            self._rebuild_index()
        if self._index.d != self._dimension:
            raise ValueError(
                "FAISS index dimension mismatch: "
                f"expected {self._dimension}, got {self._index.d}"
            )
        if self._index.ntotal != len(self._records):
            raise ValueError(
                "Inconsistent persisted vector index state: "
                f"FAISS index has {self._index.ntotal} vectors "
                f"but records file has {len(self._records)}."
            )

    def _save(self) -> None:
        if self._records_path is None:
            return
        self._records_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "doc_id": record.doc_id,
                "node_id": record.node_id,
                "chunk_id": record.chunk_id,
                "embedding": list(record.embedding),
                "text": record.text,
                "breadcrumb": list(record.breadcrumb),
            }
            for record in self._records
        ]
        if not payload:
            self._records_path.unlink(missing_ok=True)
            if self._index_path is not None:
                self._index_path.unlink(missing_ok=True)
            return
        self._records_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self._index is None or self._index_path is None:
            return
        if self._index.ntotal == 0:
            self._index_path.unlink(missing_ok=True)
            return
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self._index_path))
