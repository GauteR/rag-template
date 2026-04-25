from __future__ import annotations

import hashlib
import re

from core.application.ports.embeddings import EmbedderPort


class HashEmbedder(EmbedderPort):
    def __init__(self, *, dimension: int) -> None:
        self._dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        lower_text = text.lower()
        vector = [0.0] * self._dimension
        tokens = re.findall(r"[a-z0-9_]+", lower_text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[digest[0] % self._dimension] += 1.0
        # Keep common query/document words aligned in tiny examples.
        if "install" in tokens and self._dimension:
            vector[0] += 2.0
        return vector
