from __future__ import annotations

from typing import Any

from core.application.ports.vector_store import VectorStorePort
from core.domain.models import SearchHit, VectorRecord


class ChromaVectorStore(VectorStorePort):
    def __init__(
        self,
        *,
        collection: Any | None = None,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "rag_template",
    ) -> None:
        self._collection = collection or self._build_collection(
            host=host,
            port=port,
            collection_name=collection_name,
        )

    def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._collection.add(
            ids=[record.chunk_id for record in records],
            embeddings=[list(record.embedding) for record in records],
            documents=[record.text for record in records],
            metadatas=[
                {
                    "doc_id": record.doc_id,
                    "node_id": record.node_id,
                    "chunk_id": record.chunk_id,
                    "breadcrumb": " > ".join(record.breadcrumb),
                }
                for record in records
            ],
        )

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def search(self, embedding: list[float], *, limit: int) -> list[SearchHit]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        return self._map_result(result)

    def count(self) -> int:
        return self._collection.count()

    def doc_ids(self) -> set[str]:
        batch_size = 1000
        total = self._collection.count()
        ids: set[str] = set()

        for offset in range(0, max(total, 1), batch_size):
            result = self._collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            metadatas: list[dict[str, str]] = result.get("metadatas") or []
            if not metadatas:
                break
            ids.update(m["doc_id"] for m in metadatas if "doc_id" in m)

        return ids

    def _build_collection(self, *, host: str, port: int, collection_name: str) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install the chroma extra with `uv sync --extra chroma`.") from exc

        client = chromadb.HttpClient(host=host, port=port)
        return client.get_or_create_collection(name=collection_name)

    def _map_result(self, result: dict[str, Any]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        embeddings = result.get("embeddings", [[]])[0]

        for chunk_id, document, metadata, distance, embedding in zip(
            ids,
            documents,
            metadatas,
            distances,
            embeddings,
            strict=True,
        ):
            hits.append(
                SearchHit(
                    record=VectorRecord(
                        doc_id=metadata["doc_id"],
                        node_id=metadata["node_id"],
                        chunk_id=metadata.get("chunk_id", chunk_id),
                        embedding=tuple(float(value) for value in embedding),
                        text=document,
                        breadcrumb=tuple(
                            metadata.get("breadcrumb", "").split(" > ")
                            if metadata.get("breadcrumb")
                            else ()
                        ),
                    ),
                    score=1.0 - float(distance),
                )
            )
        return hits
