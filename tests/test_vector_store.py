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
