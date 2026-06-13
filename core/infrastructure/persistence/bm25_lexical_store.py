from __future__ import annotations

import json
import logging
from pathlib import Path

from core.application.ports.lexical_store import LexicalStorePort
from core.domain.models import SearchHit, VectorRecord

logger = logging.getLogger(__name__)


class Bm25LexicalStore(LexicalStorePort):
    def __init__(self, *, path: Path) -> None:
        self._path = path
        self._records: list[dict[str, str]] = []
        self._load()

    def index_document(self, *, doc_id: str, records: list[tuple[str, str, str]]) -> None:
        self.delete_document(doc_id)
        for node_id, chunk_id, text in records:
            self._records.append(
                {
                    "doc_id": doc_id,
                    "node_id": node_id,
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )
        self._save()

    def delete_document(self, doc_id: str) -> None:
        self._records = [record for record in self._records if record["doc_id"] != doc_id]
        self._save()

    def search(self, *, query: str, limit: int, doc_id: str | None = None) -> list[SearchHit]:
        candidates = self._records
        if doc_id is not None:
            candidates = [record for record in candidates if record["doc_id"] == doc_id]
        if not candidates:
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "Install rank-bm25 for hybrid search: uv sync --extra hybrid"
            ) from exc

        tokenized_corpus = [record["text"].lower().split() for record in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(
            zip(scores, candidates, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )[:limit]

        hits: list[SearchHit] = []
        for score, record in ranked:
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    record=VectorRecord(
                        doc_id=record["doc_id"],
                        node_id=record["node_id"],
                        chunk_id=record["chunk_id"],
                        embedding=(),
                        text=record["text"],
                        breadcrumb=(),
                    ),
                    score=float(score),
                )
            )
        return hits

    def _load(self) -> None:
        if not self._path.exists():
            return
        self._records = json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._records:
            self._path.unlink(missing_ok=True)
            return
        self._path.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
