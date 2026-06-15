from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "echo"
    llm_routing_provider: str | None = None
    llm_synthesis_provider: str | None = None
    embedding_provider: str = "hash"
    embedding_dimension: int = 8

    vector_store_provider: str = "faiss"
    section_store_provider: str = "json"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "rag_template"

    enable_llm_noise_filter: bool = False
    enable_llm_reranker: bool = False
    enable_llamaparse: bool = True
    enable_pdf_indexing: bool = True
    enable_benchmark_judge: bool = False
    enable_index_admin: bool = False
    enable_query_tracing: bool = False
    enable_streaming_query: bool = False
    enable_hybrid_search: bool = False
    enable_public_health: bool = True

    index_dir: Path = Field(default=Path(".index"))
    max_upload_mb: int = 5
    max_query_chars: int = 4000
    api_key: str | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-haiku-latest"
    llama_cloud_api_key: str | None = None

    @property
    def routing_provider(self) -> str:
        return self.llm_routing_provider or self.llm_provider

    @property
    def synthesis_provider(self) -> str:
        return self.llm_synthesis_provider or self.llm_provider

    @property
    def faiss_index_path(self) -> Path:
        return self.index_dir / "vectors.faiss"

    def validate_provider_ids(
        self,
        *,
        llm_provider_ids: set[str],
        embedding_provider_ids: set[str],
        vector_store_provider_ids: set[str] | None = None,
        section_store_provider_ids: set[str] | None = None,
    ) -> None:
        for provider in {self.routing_provider, self.synthesis_provider}:
            if provider not in llm_provider_ids:
                raise ValueError(f"Unknown LLM provider: {provider}")
        if self.embedding_provider not in embedding_provider_ids:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")
        if vector_store_provider_ids is not None:
            if self.vector_store_provider not in vector_store_provider_ids:
                raise ValueError(f"Unknown vector store provider: {self.vector_store_provider}")
        if section_store_provider_ids is not None:
            if self.section_store_provider not in section_store_provider_ids:
                raise ValueError(f"Unknown section store provider: {self.section_store_provider}")

    def validate_provider_configuration(self) -> None:
        if self.embedding_dimension < 1:
            raise ValueError("EMBEDDING_DIMENSION must be greater than 0")
        active_llm_providers = {self.routing_provider, self.synthesis_provider}
        if "anthropic" in active_llm_providers and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for anthropic provider")
        if "openai_compatible" in active_llm_providers and not self.openai_base_url:
            raise ValueError("OPENAI_BASE_URL is required for openai_compatible provider")
        if "openai_compatible" in active_llm_providers and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for openai_compatible provider")
        if self.embedding_provider == "openai_compatible" and not self.openai_base_url:
            raise ValueError("OPENAI_BASE_URL is required for openai_compatible embeddings")
        if self.embedding_provider == "openai_compatible" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for openai_compatible embeddings")
        if "ollama" in active_llm_providers and not self.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL is required for ollama provider")
        if self.embedding_provider == "ollama" and not self.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL is required for ollama embeddings")

    def validate_embedding_dimension(self, *, actual_dimension: int) -> None:
        if actual_dimension != self.embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"EMBEDDING_DIMENSION={self.embedding_dimension}, "
                f"provider returned {actual_dimension}"
            )
