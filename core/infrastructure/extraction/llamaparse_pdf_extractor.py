from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from core.application.ports.pdf_extractor import PdfExtractorPort


class LlamaParsePdfExtractor(PdfExtractorPort):
    def __init__(self, *, api_key: str | None) -> None:
        self._api_key = api_key

    def extract_markdown(self, *, filename: str, content: bytes) -> str:
        if not content:
            raise ValueError("PDF payload is empty.")

        try:
            reader = PdfReader(BytesIO(content))
        except Exception as exc:
            raise ValueError("Invalid PDF payload.") from exc

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        title = Path(filename).stem.strip() or "document"
        page_markdown: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            page_markdown.append(f"## Page {page_number}\n\n{text}")

        if not page_markdown:
            page_markdown.append("## Page 1\n\n_No extractable text found in PDF._")

        return f"# {title}\n\n" + "\n\n".join(page_markdown)
