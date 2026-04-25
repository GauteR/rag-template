from __future__ import annotations

from typing import Protocol


class EmbedderPort(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for each input text."""
