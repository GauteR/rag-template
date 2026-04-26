import json
from pathlib import Path

import pytest

from core.application.benchmarking.judge import LlmBenchmarkJudge, _parse_score
from core.application.benchmarking.models import BenchmarkQuestion, ModelProfile
from core.application.benchmarking.runner import BenchmarkRunner
from core.application.ports.llm import LlmPort
from core.domain.models import Section

_BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"


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


def test_benchmark_two_profiles_identical_retrieval() -> None:
    """Two equivalent mock profiles must produce identical retrieved_node_ids."""
    profile_a = ModelProfile(
        name="mock-a",
        llm_routing_provider="echo",
        llm_synthesis_provider="echo",
        embedding_provider="hash",
    )
    profile_b = ModelProfile(
        name="mock-b",
        llm_routing_provider="echo",
        llm_synthesis_provider="echo",
        embedding_provider="hash",
    )
    questions = [
        BenchmarkQuestion(id="q1", question="What is intro?", expected_node_ids=("manual:n1",)),
        BenchmarkQuestion(id="q2", question="How to install?", expected_node_ids=("manual:n1",)),
    ]

    runner = BenchmarkRunner(query_use_case_factory=lambda _profile: StaticQueryUseCase())
    result = runner.run(
        profiles=[profile_a, profile_b],
        questions=questions,
        k_recall=10,
        k_candidates=5,
        k_final=1,
    )

    assert len(result.rows) == 4  # 2 profiles x 2 questions

    profile_names = {row.profile_name for row in result.rows}
    assert profile_names == {"mock-a", "mock-b"}

    # Both profiles must retrieve the same node IDs for each question.
    for question in questions:
        q_rows = [r for r in result.rows if r.question_id == question.id]
        assert len(q_rows) == 2
        ids_a = next(r.retrieved_node_ids for r in q_rows if r.profile_name == "mock-a")
        ids_b = next(r.retrieved_node_ids for r in q_rows if r.profile_name == "mock-b")
        assert ids_a == ids_b

    # Both profiles must record a hit when the expected node is retrieved.
    assert all(row.hit_at_k_final is True for row in result.rows)


def test_benchmark_two_profiles_csv_contains_both_profiles(tmp_path: Path) -> None:
    """The CSV artifact must contain rows for every profile."""
    runner = BenchmarkRunner(query_use_case_factory=lambda _profile: StaticQueryUseCase())
    result = runner.run(
        profiles=[
            ModelProfile(
                name="mock-a",
                llm_routing_provider="echo",
                llm_synthesis_provider="echo",
                embedding_provider="hash",
            ),
            ModelProfile(
                name="mock-b",
                llm_routing_provider="echo",
                llm_synthesis_provider="echo",
                embedding_provider="hash",
            ),
        ],
        questions=[BenchmarkQuestion(id="q1", question="What is intro?")],
        k_recall=10,
        k_candidates=5,
        k_final=1,
    )

    artifacts = runner.write_artifacts(result=result, output_dir=tmp_path)
    csv_text = artifacts.csv_path.read_text(encoding="utf-8")

    assert "mock-a" in csv_text
    assert "mock-b" in csv_text

    json_data = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    json_profile_names = {row["profile_name"] for row in json_data["rows"]}
    assert json_profile_names == {"mock-a", "mock-b"}


def test_benchmark_yaml_configs_are_valid() -> None:
    """Verify that the example YAML configs parse without errors."""
    yaml = pytest.importorskip("yaml")

    matrix_path = _BENCHMARKS_DIR / "model_matrix.yaml"
    questions_path = _BENCHMARKS_DIR / "questions.yaml"

    assert matrix_path.exists(), "benchmarks/model_matrix.yaml must exist"
    assert questions_path.exists(), "benchmarks/questions.yaml must exist"

    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    assert len(matrix["profiles"]) >= 2, "model_matrix.yaml must define at least two profiles"

    questions = yaml.safe_load(questions_path.read_text(encoding="utf-8"))
    assert len(questions["questions"]) >= 1, "questions.yaml must define at least one question"


def test_benchmark_run_matrix_script(tmp_path: Path) -> None:
    """run_matrix.py must produce JSON and CSV artifacts for both profiles."""
    import importlib.util

    pytest.importorskip("yaml")  # ensures pyyaml is available

    script_path = _BENCHMARKS_DIR / "run_matrix.py"
    spec = importlib.util.spec_from_file_location("run_matrix", script_path)
    run_matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_matrix)

    run_matrix.main(
        [
            "--config",
            str(_BENCHMARKS_DIR / "model_matrix.yaml"),
            "--questions",
            str(_BENCHMARKS_DIR / "questions.yaml"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    artifacts = list(tmp_path.glob("benchmark-*.json"))
    assert artifacts, "run_matrix.py must write a JSON artifact"
    data = json.loads(artifacts[0].read_text(encoding="utf-8"))
    profile_names = {row["profile_name"] for row in data["rows"]}
    assert len(profile_names) >= 2, "JSON artifact must contain at least two profiles"

    csv_artifacts = list(tmp_path.glob("benchmark-*.csv"))
    assert csv_artifacts, "run_matrix.py must write a CSV artifact"
    csv_text = csv_artifacts[0].read_text(encoding="utf-8")
    for name in profile_names:
        assert name in csv_text, f"CSV artifact must contain rows for profile '{name}'"


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


def test_benchmark_judge_enabled_calls_llm_once_per_question() -> None:
    """When a judge is wired in, the underlying LLM must be called once per question."""
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

    judge = LlmBenchmarkJudge(llm=SpyLlm())
    runner = BenchmarkRunner(
        query_use_case_factory=lambda _profile: StaticQueryUseCase(),
        judge=judge,
    )
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

    # SpyLlm.synthesize must be called exactly once (one question, one judge invocation).
    assert len(call_log) == 1


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
