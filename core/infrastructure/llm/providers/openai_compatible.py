from __future__ import annotations

import json

import httpx

from core.application.ports.llm import LlmPort
from core.domain.models import Section
from core.infrastructure.llm.json_utils import parse_json_object


class OpenAiCompatibleLlm(LlmPort):
    def __init__(self, *, base_url: str, api_key: str | None, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def filter_noise(self, *, sections: list[Section]) -> set[str]:
        prompt = {
            "sections": [
                {
                    "node_id": section.node_id,
                    "breadcrumb": " > ".join(section.breadcrumb),
                    "text": section.text[:500],
                }
                for section in sections
            ],
            "instruction": "Return JSON only with noisy_node_ids for boilerplate sections.",
        }
        content = self._chat(json.dumps(prompt))
        parsed = parse_json_object(content)
        node_ids = parsed.get("noisy_node_ids", [])
        return {node_id for node_id in node_ids if isinstance(node_id, str)}

    def rerank(self, *, question: str, candidates: list[Section], k_final: int) -> list[str]:
        prompt = {
            "question": question,
            "candidates": [
                {"node_id": section.node_id, "breadcrumb": " > ".join(section.breadcrumb)}
                for section in candidates
            ],
            "instruction": f"Return JSON with node_ids, top {k_final}, most relevant first.",
        }
        content = self._chat(json.dumps(prompt))
        parsed = parse_json_object(content)
        node_ids = parsed.get("node_ids", [])
        return [node_id for node_id in node_ids if isinstance(node_id, str)][:k_final]

    def synthesize(self, *, question: str, sections: list[Section]) -> str:
        context = "\n\n".join(
            f"[{section.node_id}] {' > '.join(section.breadcrumb)}\n{section.text}"
            for section in sections
        )
        return self._chat(
            f"Answer the question from the context.\nQuestion: {question}\n\n{context}"
        )

    def _chat(self, content: str) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        with httpx.Client(timeout=60, headers=headers) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
