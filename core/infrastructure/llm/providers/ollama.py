from __future__ import annotations

from core.infrastructure.llm.providers.openai_compatible import OpenAiCompatibleLlm


class OllamaLlm(OpenAiCompatibleLlm):
    def __init__(self, *, base_url: str, model: str) -> None:
        super().__init__(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=None,
            model=model,
        )
