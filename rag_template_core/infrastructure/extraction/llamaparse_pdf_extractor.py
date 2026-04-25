from __future__ import annotations

from rag_template_core.application.ports.pdf_extractor import PdfExtractorPort


class LlamaParsePdfExtractor(PdfExtractorPort):
    def __init__(self, *, api_key: str | None) -> None:
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY is required when ENABLE_LLAMAPARSE=true")
        self._api_key = api_key

    def extract_markdown(self, *, filename: str, content: bytes) -> str:
        del filename, content
        raise NotImplementedError(
            "Install the llamaparse extra and wire the official client "
            "before enabling PDF indexing."
        )
