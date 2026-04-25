from __future__ import annotations

from typing import Protocol


class PdfExtractorPort(Protocol):
    def extract_markdown(self, *, filename: str, content: bytes) -> str:
        """Convert a PDF file to Markdown text."""
