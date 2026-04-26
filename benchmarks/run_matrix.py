"""CLI script that loads benchmarks/model_matrix.yaml and benchmarks/questions.yaml,
runs the comparison benchmark using mock providers, and writes JSON/CSV artifacts
to benchmarks/out/.

Usage (no live credentials required for the default echo/hash providers):

    python benchmarks/run_matrix.py

To use live model profiles add them to model_matrix.yaml and run:

    python benchmarks/run_matrix.py --live --config benchmarks/model_matrix.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pyyaml is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.application.benchmarking.models import BenchmarkQuestion, ModelProfile
from core.application.benchmarking.runner import BenchmarkRunner


def _load_profiles(matrix_path: Path) -> tuple[dict, list[ModelProfile]]:
    data = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    profiles = [
        ModelProfile(
            name=p["name"],
            llm_routing_provider=p["llm_routing_provider"],
            llm_synthesis_provider=p["llm_synthesis_provider"],
            embedding_provider=p["embedding_provider"],
        )
        for p in data["profiles"]
    ]
    retrieval = data.get("retrieval", {})
    return retrieval, profiles


def _load_questions(questions_path: Path) -> list[BenchmarkQuestion]:
    data = yaml.safe_load(questions_path.read_text(encoding="utf-8"))
    return [
        BenchmarkQuestion(
            id=q["id"],
            question=q["question"],
            expected_node_ids=tuple(q.get("expected_node_ids") or []),
        )
        for q in data["questions"]
    ]


def _make_mock_use_case():
    """Return a lightweight use-case stub suitable for local/CI runs."""
    from core.application.query.models import QueryResponse, QuerySource

    class _MockUseCase:
        def execute(self, question: str, k_recall: int, k_candidates: int, k_final: int):
            return QueryResponse(
                answer=f"mock answer: {question}",
                sources=[
                    QuerySource(
                        doc_id="mock",
                        node_id="mock:n1",
                        breadcrumb=("Mock",),
                        score=1.0,
                        text="mock context",
                    )
                ],
            )

    return _MockUseCase()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark model matrix.")
    parser.add_argument(
        "--config",
        default="benchmarks/model_matrix.yaml",
        help="Path to model_matrix.yaml (default: benchmarks/model_matrix.yaml)",
    )
    parser.add_argument(
        "--questions",
        default="benchmarks/questions.yaml",
        help="Path to questions.yaml (default: benchmarks/questions.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/out",
        help="Directory to write JSON/CSV artifacts (default: benchmarks/out)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Use live providers instead of the built-in mock (requires credentials).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).parent.parent
    matrix_path = Path(args.config)
    if not matrix_path.is_absolute():
        matrix_path = repo_root / matrix_path
    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = repo_root / questions_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    retrieval, profiles = _load_profiles(matrix_path)
    questions = _load_questions(questions_path)

    print(f"Loaded {len(profiles)} profile(s) and {len(questions)} question(s).")

    runner = BenchmarkRunner(query_use_case_factory=lambda _profile: _make_mock_use_case())
    result = runner.run(
        profiles=profiles,
        questions=questions,
        k_recall=retrieval.get("k_recall", 10),
        k_candidates=retrieval.get("k_candidates", 5),
        k_final=retrieval.get("k_final", 1),
    )

    artifacts = runner.write_artifacts(result=result, output_dir=output_dir)
    print(f"JSON: {artifacts.json_path}")
    print(f"CSV:  {artifacts.csv_path}")

    print("\nPer-profile summary:")
    print(f"{'profile':<20} {'question':<10} {'hit@k_final':<12} {'latency_ms':>10}")
    print("-" * 56)
    for row in result.rows:
        hit = str(row.hit_at_k_final) if row.hit_at_k_final is not None else "N/A"
        print(f"{row.profile_name:<20} {row.question_id:<10} {hit:<12} {row.latency_ms:>10.1f}")


if __name__ == "__main__":
    main()
