from core.domain.models import SearchHit, VectorRecord
from core.infrastructure.persistence.chroma_vector_store import ChromaVectorStore


class FakeCollection:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
        self._records: list[dict[str, object]] = []

    def add(self, **kwargs) -> None:
        self.add_calls.append(kwargs)
        ids = kwargs.get("ids", [])
        metadatas = kwargs.get("metadatas", [])
        for chunk_id, metadata in zip(ids, metadatas, strict=True):
            self._records.append({"id": chunk_id, "metadata": metadata})

    def delete(self, **kwargs) -> None:
        self.delete_calls.append(kwargs)
        where = kwargs.get("where", {})
        doc_id = where.get("doc_id")
        if doc_id:
            self._records = [r for r in self._records if r["metadata"]["doc_id"] != doc_id]

    def count(self) -> int:
        return len(self._records)

    def get(self, **kwargs) -> dict[str, object]:
        limit = kwargs.get("limit", len(self._records))
        offset = kwargs.get("offset", 0)
        page = self._records[offset : offset + limit]
        return {"metadatas": [r["metadata"] for r in page]}

    def query(self, **kwargs) -> dict[str, object]:
        assert kwargs["query_embeddings"] == [[1.0, 0.0]]
        assert kwargs["n_results"] == 2
        return {
            "ids": [["doc:n1:c1"]],
            "distances": [[0.25]],
            "documents": [["chunk text"]],
            "metadatas": [
                [
                    {
                        "doc_id": "doc",
                        "node_id": "doc:n1",
                        "chunk_id": "doc:n1:c1",
                        "breadcrumb": "Root > Child",
                    }
                ]
            ],
            "embeddings": [[[1.0, 0.0]]],
        }


def test_chroma_vector_store_adds_records_with_proxy_pointer_metadata() -> None:
    collection = FakeCollection()
    store = ChromaVectorStore(collection=collection)

    store.add(
        [
            VectorRecord(
                doc_id="doc",
                node_id="doc:n1",
                chunk_id="doc:n1:c1",
                embedding=(1.0, 0.0),
                text="chunk text",
                breadcrumb=("Root", "Child"),
            )
        ]
    )

    call = collection.add_calls[0]
    assert call["ids"] == ["doc:n1:c1"]
    assert call["documents"] == ["chunk text"]
    assert call["metadatas"][0]["node_id"] == "doc:n1"
    assert call["metadatas"][0]["breadcrumb"] == "Root > Child"


def test_chroma_vector_store_deletes_by_doc_id_and_maps_search_hits() -> None:
    collection = FakeCollection()
    store = ChromaVectorStore(collection=collection)

    store.delete_document("doc")
    hits = store.search([1.0, 0.0], limit=2)

    assert collection.delete_calls[0]["where"] == {"doc_id": "doc"}
    assert hits == [
        SearchHit(
            record=VectorRecord(
                doc_id="doc",
                node_id="doc:n1",
                chunk_id="doc:n1:c1",
                embedding=(1.0, 0.0),
                text="chunk text",
                breadcrumb=("Root", "Child"),
            ),
            score=0.75,
        )
    ]


def test_chroma_vector_store_count_and_doc_ids() -> None:
    collection = FakeCollection()
    store = ChromaVectorStore(collection=collection)

    assert store.count() == 0
    assert store.doc_ids() == set()

    store.add(
        [
            VectorRecord(
                doc_id="doc-a",
                node_id="doc-a:n1",
                chunk_id="doc-a:n1:c0",
                embedding=(1.0, 0.0),
                text="first chunk",
                breadcrumb=("A",),
            ),
            VectorRecord(
                doc_id="doc-a",
                node_id="doc-a:n2",
                chunk_id="doc-a:n2:c0",
                embedding=(0.0, 1.0),
                text="second chunk",
                breadcrumb=("A", "Sub"),
            ),
            VectorRecord(
                doc_id="doc-b",
                node_id="doc-b:n1",
                chunk_id="doc-b:n1:c0",
                embedding=(1.0, 1.0),
                text="other doc",
                breadcrumb=("B",),
            ),
        ]
    )

    assert store.count() == 3
    assert store.doc_ids() == {"doc-a", "doc-b"}

    store.delete_document("doc-a")
    assert store.count() == 1
    assert store.doc_ids() == {"doc-b"}
