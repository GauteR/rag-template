from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import gmtime, perf_counter, strftime

from rag_template_core.application.benchmarking.models import (
    BenchmarkArtifacts,
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkRow,
    ModelProfile,
)


class BenchmarkRunner:
    def __init__(self, *, query_use_case_factory: Callable[[ModelProfile], object]) -> None:
        self._query_use_case_factory = query_use_case_factory

    def run(
        self,
        *,
        profiles: list[ModelProfile],
        questions: list[BenchmarkQuestion],
        k_recall: int,
        k_candidates: int,
        k_final: int,
    ) -> BenchmarkResult:
        rows: list[BenchmarkRow] = []
        for profile in profiles:
            query_use_case = self._query_use_case_factory(profile)
            for question in questions:
                start = perf_counter()
                response = query_use_case.execute(
                    question=question.question,
                    k_recall=k_recall,
                    k_candidates=k_candidates,
                    k_final=k_final,
                )
                latency_ms = (perf_counter() - start) * 1_000
                retrieved_node_ids = tuple(source.node_id for source in response.sources)
                rows.append(
                    BenchmarkRow(
                        profile_name=profile.name,
                        question_id=question.id,
                        latency_ms=latency_ms,
                        retrieved_node_ids=retrieved_node_ids,
                        hit_at_k_final=self._hit_at_k(
                            question.expected_node_ids, retrieved_node_ids
                        ),
                    )
                )
        return BenchmarkResult(rows=rows)

    def _hit_at_k(
        self,
        expected_node_ids: tuple[str, ...],
        retrieved_node_ids: tuple[str, ...],
    ) -> bool | None:
        if not expected_node_ids:
            return None
        return bool(set(expected_node_ids).intersection(retrieved_node_ids))

    def write_artifacts(self, *, result: BenchmarkResult, output_dir: Path) -> BenchmarkArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = strftime("%Y%m%dT%H%M%SZ", gmtime())
        json_path = output_dir / f"benchmark-{timestamp}.json"
        csv_path = output_dir / f"benchmark-{timestamp}.csv"

        rows = [asdict(row) for row in result.rows]
        json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "profile_name",
                    "question_id",
                    "latency_ms",
                    "retrieved_node_ids",
                    "hit_at_k_final",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return BenchmarkArtifacts(json_path=json_path, csv_path=csv_path)
