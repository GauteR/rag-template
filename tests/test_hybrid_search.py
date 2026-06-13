import pytest

from app.container import AppContainer
from core.config.settings import Settings


def test_hybrid_search_combines_vector_and_lexical(tmp_path) -> None:
    pytest.importorskip("rank_bm25")

    settings = Settings(index_dir=tmp_path, enable_hybrid_search=True)
    container = AppContainer(settings=settings)
    use_case = container.index_markdown_use_case()
    use_case.execute(
        doc_id="manual",
        markdown="# Intro\nWelcome\n\n## Install\nInstall with uv sync",
    )

    response = container.query_use_case().execute(
        question="install uv sync",
        k_recall=10,
        k_candidates=5,
        k_final=1,
    )

    assert response.sources
    assert response.sources[0].node_id == "manual:n2"
