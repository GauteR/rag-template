from __future__ import annotations

import httpx

from core.application.ports.llm import LlmPort
from core.domain.models import Section
from core.infrastructure.llm.json_utils import parse_json_object


class AnthropicLlm(LlmPort):
    def __init__(self, *, api_key: str | None, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def filter_noise(self, *, sections: list[Section]) -> set[str]:
        prompt = "\n".join(
            [
                'Return JSON only: {"noisy_node_ids": [...]} for boilerplate sections.',
                *[
                    f"- {section.node_id}: {' > '.join(section.breadcrumb)}\n{section.text[:500]}"
                    for section in sections
                ],
            ]
        )
        content = self._message(prompt)
        parsed = parse_json_object(content)
        node_ids = parsed.get("noisy_node_ids", [])
        return {node_id for node_id in node_ids if isinstance(node_id, str)}

    def rerank(self, *, question: str, candidates: list[Section], k_final: int) -> list[str]:
        # Keep provider-specific JSON handling inside the adapter.
        prompt = "\n".join(
            [
                f"Question: {question}",
                f'Return JSON only: {{"node_ids": [...]}} with top {k_final} node IDs.',
                *[
                    f"- {section.node_id}: {' > '.join(section.breadcrumb)}"
                    for section in candidates
                ],
            ]
        )
        content = self._message(prompt)
        parsed = parse_json_object(content)
        node_ids = parsed.get("node_ids", [])
        return [node_id for node_id in node_ids if isinstance(node_id, str)][:k_final]

    def synthesize(self, *, question: str, sections: list[Section]) -> str:
        context = "\n\n".join(section.text for section in sections)
        return self._message(f"Answer from context.\nQuestion: {question}\n\n{context}")

    def _message(self, prompt: str) -> str:
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for anthropic provider")
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self._model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            parts = response.json()["content"]
            return "".join(part.get("text", "") for part in parts)
