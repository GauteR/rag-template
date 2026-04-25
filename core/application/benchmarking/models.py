from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    name: str
    llm_routing_provider: str
    llm_synthesis_provider: str
    embedding_provider: str


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    question: str
    expected_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkRow:
    profile_name: str
    question_id: str
    latency_ms: float
    retrieved_node_ids: tuple[str, ...]
    hit_at_k_final: bool | None


@dataclass(frozen=True)
class BenchmarkResult:
    rows: list[BenchmarkRow]


@dataclass(frozen=True)
class BenchmarkArtifacts:
    json_path: Path
    csv_path: Path
