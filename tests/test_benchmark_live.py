import importlib.util
import json
from pathlib import Path

import pytest

_BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"


def test_seed_index_and_live_benchmark(tmp_path: Path) -> None:
    pytest.importorskip("yaml")

    seed_spec = importlib.util.spec_from_file_location(
        "seed_index", _BENCHMARKS_DIR / "seed_index.py"
    )
    seed_index = importlib.util.module_from_spec(seed_spec)
    seed_spec.loader.exec_module(seed_index)

    index_dir = tmp_path / "benchmark-index"
    fixtures_dir = _BENCHMARKS_DIR / "fixtures" / "docs"
    node_ids = seed_index.seed_index(index_dir=index_dir, fixtures_dir=fixtures_dir)
    assert "install" in node_ids
    assert node_ids["install"]

    run_spec = importlib.util.spec_from_file_location(
        "run_matrix", _BENCHMARKS_DIR / "run_matrix.py"
    )
    run_matrix = importlib.util.module_from_spec(run_spec)
    run_spec.loader.exec_module(run_matrix)

    output_dir = tmp_path / "out"
    run_matrix.main(
        [
            "--live",
            "--index-dir",
            str(index_dir),
            "--config",
            str(_BENCHMARKS_DIR / "model_matrix.yaml"),
            "--questions",
            str(_BENCHMARKS_DIR / "questions.yaml"),
            "--output-dir",
            str(output_dir),
        ]
    )

    artifacts = list(output_dir.glob("benchmark-*.json"))
    assert artifacts
    data = json.loads(artifacts[0].read_text(encoding="utf-8"))
    hits = [row["hit_at_k_final"] for row in data["rows"] if row["question_id"] == "q1"]
    assert any(hit is True for hit in hits)
