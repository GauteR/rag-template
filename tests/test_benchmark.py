from core.application.benchmarking.models import BenchmarkQuestion, ModelProfile
from core.application.benchmarking.runner import BenchmarkRunner


class StaticQueryUseCase:
    def execute(self, question: str, k_recall: int, k_candidates: int, k_final: int):
        from core.application.query.models import QueryResponse, QuerySource

        return QueryResponse(
            answer=f"answer: {question}",
            sources=[
                QuerySource(
                    doc_id="manual",
                    node_id="manual:n1",
                    breadcrumb=("Intro",),
                    score=1.0,
                    text="context",
                )
            ],
        )


def test_benchmark_runner_records_latency_and_hit_at_k() -> None:
    runner = BenchmarkRunner(query_use_case_factory=lambda _profile: StaticQueryUseCase())

    result = runner.run(
        profiles=[
            ModelProfile(
                name="mock-a",
                llm_routing_provider="echo",
                llm_synthesis_provider="echo",
                embedding_provider="hash",
            )
        ],
        questions=[
            BenchmarkQuestion(id="q1", question="What is intro?", expected_node_ids=("manual:n1",))
        ],
        k_recall=10,
        k_candidates=5,
        k_final=1,
    )

    assert result.rows[0].profile_name == "mock-a"
    assert result.rows[0].hit_at_k_final is True
    assert result.rows[0].latency_ms >= 0


def test_benchmark_runner_writes_json_and_csv_artifacts(tmp_path) -> None:
    runner = BenchmarkRunner(query_use_case_factory=lambda _profile: StaticQueryUseCase())
    result = runner.run(
        profiles=[
            ModelProfile(
                name="mock-a",
                llm_routing_provider="echo",
                llm_synthesis_provider="echo",
                embedding_provider="hash",
            )
        ],
        questions=[BenchmarkQuestion(id="q1", question="What is intro?")],
        k_recall=10,
        k_candidates=5,
        k_final=1,
    )

    artifacts = runner.write_artifacts(result=result, output_dir=tmp_path)

    assert artifacts.json_path.exists()
    assert artifacts.csv_path.exists()
    assert "profile_name,question_id,latency_ms,retrieved_node_ids,hit_at_k_final" in (
        artifacts.csv_path.read_text(encoding="utf-8")
    )
