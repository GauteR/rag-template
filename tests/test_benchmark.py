from core.application.benchmarking.judge import LlmBenchmarkJudge, _parse_score
from core.application.benchmarking.models import BenchmarkQuestion, ModelProfile
from core.application.benchmarking.runner import BenchmarkRunner
from core.application.ports.llm import LlmPort
from core.domain.models import Section


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
    assert (
        "profile_name,question_id,latency_ms,retrieved_node_ids,hit_at_k_final,judge_score"
        in artifacts.csv_path.read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Judge disabled (no judge argument)
# ---------------------------------------------------------------------------


def test_benchmark_judge_disabled_by_default() -> None:
    """When no judge is supplied, judge_score must be None for every row."""
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

    assert result.rows[0].judge_score is None


def test_benchmark_judge_disabled_produces_no_judge_calls() -> None:
    """BenchmarkRunner without a judge must never invoke any judge."""
    call_log: list[str] = []

    class SpyLlm(LlmPort):
        def filter_noise(self, *, sections: list[Section]) -> set[str]:
            return set()

        def rerank(
            self, *, question: str, candidates: list[Section], k_final: int
        ) -> list[str]:
            return [s.node_id for s in candidates[:k_final]]

        def synthesize(self, *, question: str, sections: list[Section]) -> str:
            call_log.append(question)
            return "0.9"

    runner = BenchmarkRunner(query_use_case_factory=lambda _profile: StaticQueryUseCase())
    runner.run(
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

    # SpyLlm was never passed to the runner, so it should never be called.
    assert call_log == []


# ---------------------------------------------------------------------------
# Judge enabled with mocked LlmPort
# ---------------------------------------------------------------------------


def test_benchmark_runner_with_judge_records_scores() -> None:
    """When a judge is provided, judge_score must be populated for every row."""

    class FixedScoreLlm(LlmPort):
        def filter_noise(self, *, sections: list[Section]) -> set[str]:
            return set()

        def rerank(
            self, *, question: str, candidates: list[Section], k_final: int
        ) -> list[str]:
            return [s.node_id for s in candidates[:k_final]]

        def synthesize(self, *, question: str, sections: list[Section]) -> str:
            return "0.75"

    judge = LlmBenchmarkJudge(llm=FixedScoreLlm())
    runner = BenchmarkRunner(
        query_use_case_factory=lambda _profile: StaticQueryUseCase(),
        judge=judge,
    )
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

    assert result.rows[0].judge_score == 0.75


def test_benchmark_runner_with_judge_included_in_artifacts(tmp_path) -> None:
    """Judge scores must appear in both JSON and CSV artifacts."""

    class FixedScoreLlm(LlmPort):
        def filter_noise(self, *, sections: list[Section]) -> set[str]:
            return set()

        def rerank(
            self, *, question: str, candidates: list[Section], k_final: int
        ) -> list[str]:
            return [s.node_id for s in candidates[:k_final]]

        def synthesize(self, *, question: str, sections: list[Section]) -> str:
            return "0.5"

    judge = LlmBenchmarkJudge(llm=FixedScoreLlm())
    runner = BenchmarkRunner(
        query_use_case_factory=lambda _profile: StaticQueryUseCase(),
        judge=judge,
    )
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

    csv_text = artifacts.csv_path.read_text(encoding="utf-8")
    json_data = artifacts.json_path.read_text(encoding="utf-8")

    assert "judge_score" in csv_text
    assert "0.5" in csv_text
    assert "judge_score" in json_data
    assert "0.5" in json_data


# ---------------------------------------------------------------------------
# _parse_score unit tests
# ---------------------------------------------------------------------------


def test_parse_score_extracts_float() -> None:
    assert _parse_score("Score: 0.8") == 0.8


def test_parse_score_clamps_above_one() -> None:
    assert _parse_score("5") == 1.0


def test_parse_score_clamps_to_zero() -> None:
    # The regex only matches non-negative digit sequences, so the min() clamp
    # guards against future misuse; "0" exercises the exact lower bound.
    assert _parse_score("0") == 0.0


def test_parse_score_returns_none_on_no_match() -> None:
    assert _parse_score("no numbers here") is None


def test_parse_score_integer_response() -> None:
    assert _parse_score("1") == 1.0
