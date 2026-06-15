"""CLI script that loads benchmarks/model_matrix.yaml and benchmarks/questions.yaml,
runs the comparison benchmark, and writes JSON/CSV artifacts to benchmarks/out/.

Usage:

    python benchmarks/run_matrix.py --mock
    python benchmarks/seed_index.py
    python benchmarks/run_matrix.py --live --index-dir benchmarks/.benchmark-index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.application.benchmarking.judge import LlmBenchmarkJudge
from core.application.benchmarking.models import BenchmarkQuestion, ModelProfile
from core.application.benchmarking.runner import BenchmarkRunner
from core.config.settings import Settings


def _load_profiles(matrix_path: Path) -> tuple[dict, list[ModelProfile]]:
    import yaml

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
    import yaml

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
        def execute(self, question: str, k_recall: int, k_candidates: int, k_final: int, **kwargs):
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


def _make_live_factory(*, index_dir: Path):
    from app.container import AppContainer

    base_settings = Settings(index_dir=index_dir, vector_store_provider="faiss")

    def factory(profile: ModelProfile):
        container = AppContainer(
            settings=base_settings.model_copy(
                update={
                    "llm_provider": profile.llm_synthesis_provider,
                    "llm_routing_provider": profile.llm_routing_provider,
                    "llm_synthesis_provider": profile.llm_synthesis_provider,
                    "embedding_provider": profile.embedding_provider,
                }
            )
        )
        return container.query_use_case

    return factory


def main(argv: list[str] | None = None) -> None:
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise ImportError("pyyaml is required: pip install pyyaml") from None

    parser = argparse.ArgumentParser(description="Run the benchmark model matrix.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock",
        dest="mode",
        action="store_const",
        const="mock",
        help="Use mock query use case (default)",
    )
    mode.add_argument(
        "--live",
        dest="mode",
        action="store_const",
        const="live",
        help="Run against the real query pipeline",
    )
    parser.set_defaults(mode="mock")
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
        "--index-dir",
        default="benchmarks/.benchmark-index",
        help="Index directory for --live runs (default: benchmarks/.benchmark-index)",
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
    index_dir = Path(args.index_dir)
    if not index_dir.is_absolute():
        index_dir = repo_root / index_dir

    retrieval, profiles = _load_profiles(matrix_path)
    questions = _load_questions(questions_path)

    print(f"Loaded {len(profiles)} profile(s) and {len(questions)} question(s).")

    settings = Settings()
    judge = None
    if settings.enable_benchmark_judge:
        from app.container import AppContainer

        judge_container = AppContainer(settings=settings)
        judge = LlmBenchmarkJudge(llm=judge_container.reranker_llm)

    if args.mode == "live":
        query_use_case_factory = _make_live_factory(index_dir=index_dir)
        print(f"Running live benchmark against index at {index_dir}")
    else:

        def mock_factory(_profile: ModelProfile):
            return _make_mock_use_case()

        query_use_case_factory = mock_factory
        print("Running mock benchmark")

    runner = BenchmarkRunner(query_use_case_factory=query_use_case_factory, judge=judge)
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
    print(f"{'profile':<20} {'question':<10} {'hit@k_final':<12} {'judge':<8} {'latency_ms':>10}")
    print("-" * 68)
    for row in result.rows:
        hit = str(row.hit_at_k_final) if row.hit_at_k_final is not None else "N/A"
        judge_score = f"{row.judge_score:.2f}" if row.judge_score is not None else "N/A"
        print(
            f"{row.profile_name:<20} {row.question_id:<10} {hit:<12} "
            f"{judge_score:<8} {row.latency_ms:>10.1f}"
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
