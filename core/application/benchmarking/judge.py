from __future__ import annotations

import re
from typing import Protocol

from core.application.ports.llm import LlmPort

_JUDGE_PROMPT_TEMPLATE = (
    "You are an impartial benchmark judge. "
    "Score the following answer to the question on a scale from 0.0 (completely wrong) "
    "to 1.0 (perfectly correct and complete).\n\n"
    "Question: {question}\n\n"
    "Answer: {answer}\n\n"
    "Reply with ONLY a number between 0.0 and 1.0."
)


class BenchmarkJudgePort(Protocol):
    def score(self, *, question: str, answer: str) -> float | None:
        """Return a numeric quality score in [0.0, 1.0], or None if scoring failed."""


class LlmBenchmarkJudge:
    """Benchmark judge backed by an LlmPort.

    The judge synthesises a score by presenting the question/answer pair to the
    LLM as a plain-text prompt and parsing the first floating-point number found
    in the response.
    """

    def __init__(self, *, llm: LlmPort) -> None:
        self._llm = llm

    def score(self, *, question: str, answer: str) -> float | None:
        prompt = _JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
        # We abuse the synthesize interface: pass the prompt as the question and
        # supply no sections so the LLM only sees the judge instruction.
        raw = self._llm.synthesize(question=prompt, sections=[])
        return _parse_score(raw)


def _parse_score(text: str) -> float | None:
    """Extract the first float in *text* and clamp it to [0.0, 1.0]."""
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, value))
