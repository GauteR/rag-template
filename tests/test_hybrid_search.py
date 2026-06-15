import pytest

from app.container import AppContainer
from core.application.query.use_case import QueryUseCase
from core.config.settings import Settings
from core.domain.models import SearchHit, VectorRecord


def test_hybrid_search_combines_vector_and_lexical(tmp_path) -> None:
    pytest.importorskip("rank_bm25")

    settings = Settings(index_dir=tmp_path, enable_hybrid_search=True)
    container = AppContainer(settings=settings)
    use_case = container.index_markdown_use_case
    use_case.execute(
        doc_id="manual",
        markdown="# Intro\nWelcome\n\n## Install\nInstall with uv sync",
    )

    response = container.query_use_case.execute(
        question="install uv sync",
        k_recall=10,
        k_candidates=5,
        k_final=1,
    )

    assert response.sources
    assert response.sources[0].node_id == "manual:n2"


def test_merge_hits_uses_reciprocal_rank_fusion() -> None:
    use_case = QueryUseCase(
        embedder=None,  # type: ignore[arg-type]
        vector_store=None,  # type: ignore[arg-type]
        section_source=None,  # type: ignore[arg-type]
        synthesis_llm=None,
        reranker_llm=None,
        enable_llm_reranker=False,
    )
    vector_hits = [
        SearchHit(
            record=VectorRecord(
                doc_id="doc",
                node_id="a",
                chunk_id="a:c1",
                embedding=(1.0,),
                text="alpha",
                breadcrumb=("A",),
            ),
            score=0.9,
        ),
        SearchHit(
            record=VectorRecord(
                doc_id="doc",
                node_id="b",
                chunk_id="b:c1",
                embedding=(0.5,),
                text="beta",
                breadcrumb=("B",),
            ),
            score=0.1,
        ),
    ]
    lexical_hits = [
        SearchHit(
            record=VectorRecord(
                doc_id="doc",
                node_id="b",
                chunk_id="b:c1",
                embedding=(),
                text="beta",
                breadcrumb=("B",),
            ),
            score=42.0,
        ),
        SearchHit(
            record=VectorRecord(
                doc_id="doc",
                node_id="c",
                chunk_id="c:c1",
                embedding=(),
                text="gamma",
                breadcrumb=("C",),
            ),
            score=10.0,
        ),
    ]

    merged = use_case._merge_hits(vector_hits, lexical_hits)

    assert [hit.record.node_id for hit in merged] == ["b", "a", "c"]
    assert merged[0].score > merged[1].score > merged[2].score


def test_hybrid_search_reports_missing_rank_bm25_dependency(tmp_path, monkeypatch) -> None:
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def missing_rank_bm25(name: str, package: str | None = None):
        if name == "rank_bm25":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", missing_rank_bm25)
    settings = Settings(index_dir=tmp_path, enable_hybrid_search=True)
    container = AppContainer(settings=settings)

    errors = container.collect_config_errors()

    assert any("rank-bm25" in error for error in errors)
