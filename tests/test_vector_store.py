import pytest

from core.domain.models import VectorRecord
from core.infrastructure.persistence.faiss_vector_store import (
    FaissVectorStore,
    normalize_vector,
)


def test_faiss_vector_store_rejects_embedding_dimension_mismatch() -> None:
    store = FaissVectorStore(dimension=2)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        store.add(
            [
                VectorRecord(
                    doc_id="doc",
                    node_id="doc:n1",
                    chunk_id="doc:n1:c1",
                    embedding=(1.0,),
                    text="content",
                    breadcrumb=("Doc",),
                )
            ]
        )


def test_normalize_vector_keeps_faiss_inner_product_equivalent_to_cosine() -> None:
    assert normalize_vector((3.0, 4.0)) == (0.6, 0.8)
    assert normalize_vector((0.0, 0.0)) == (0.0, 0.0)


def test_faiss_vector_store_persists_records_across_restarts(tmp_path) -> None:
    index_path = tmp_path / "vectors.faiss"
    store = FaissVectorStore(dimension=2, index_path=index_path)
    store.add(
        [
            VectorRecord(
                doc_id="manual",
                node_id="manual:n1",
                chunk_id="manual:n1:c1",
                embedding=(1.0, 0.0),
                text="install",
                breadcrumb=("Manual", "Install"),
            )
        ]
    )

    restarted_store = FaissVectorStore(dimension=2, index_path=index_path)
    hits = restarted_store.search([1.0, 0.0], limit=1)

    assert restarted_store.count() == 1
    assert hits[0].record.node_id == "manual:n1"
    assert (tmp_path / "vectors.records.json").exists()


def test_faiss_vector_store_raises_clear_error_for_corrupt_records_file(tmp_path) -> None:
    index_path = tmp_path / "vectors.faiss"
    (tmp_path / "vectors.records.json").write_text("{not valid json}", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to load persisted vector records"):
        FaissVectorStore(dimension=2, index_path=index_path)


def test_faiss_vector_store_raises_for_index_without_records_file(tmp_path) -> None:
    index_path = tmp_path / "vectors.faiss"
    index_path.write_bytes(b"placeholder")

    with pytest.raises(
        ValueError,
        match="Inconsistent persisted vector index state",
    ):
        FaissVectorStore(dimension=2, index_path=index_path)
