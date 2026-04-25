from __future__ import annotations

import hashlib
import tempfile
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from core.application.ports.pdf_extractor import PdfExtractorPort


class LlamaParsePdfExtractor(PdfExtractorPort):
    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key

    def extract_markdown(self, *, filename: str, content: bytes) -> str:
        if not content:
            raise ValueError("PDF payload is empty.")
        if self._api_key:
            return self._extract_with_llamaparse(filename=filename, content=content)
        return self._extract_with_pypdf(filename=filename, content=content)

    def _extract_with_llamaparse(self, *, filename: str, content: bytes) -> str:
        try:
            from llama_parse import LlamaParse
        except ImportError as exc:
            raise ValueError(
                "Rich PDF extraction requires the llamaparse extra (`uv sync --extra llamaparse`)."
            ) from exc

        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(content)
            tmp.flush()
            documents = LlamaParse(
                api_key=self._api_key,
                result_type="markdown",
            ).load_data(tmp.name)

        markdown = "\n\n".join(
            doc.text.strip()
            for doc in documents
            if getattr(doc, "text", None) and doc.text.strip()
        ).strip()
        if not markdown:
            raise ValueError("No extractable content found in PDF.")
        return markdown

    def _extract_with_pypdf(self, *, filename: str, content: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(content))
        except (PdfReadError, ValueError) as exc:
            raise ValueError("Invalid PDF payload.") from exc

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        title = Path(filename).stem.strip() or f"document-{hashlib.sha1(content).hexdigest()[:8]}"
        page_markdown: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            page_markdown.append(f"## Page {page_number}\n\n{text}")

        if not page_markdown:
            page_markdown.append(
                f"_No extractable text found across {len(reader.pages)} page(s) in PDF._"
            )

        return f"# {title}\n\n" + "\n\n".join(page_markdown)
