"""
Application settings — loaded from environment variables / .env file.
All settings are validated at startup; missing required values raise an error
before any model weights are loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Embedding & reranking ─────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = Field(default=64, ge=1, le=512)
    embedding_device: str = "cpu"  # "cuda" if GPU available

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64, le=8192)

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=512)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    hybrid_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    top_k_retrieval: int = Field(default=20, ge=1, le=200)
    top_k_final: int = Field(default=5, ge=1, le=50)

    # ── Storage ───────────────────────────────────────────────────────────────
    index_dir: Path = Path("data/processed")

    # ── Caching ───────────────────────────────────────────────────────────────
    redis_url: str | None = None
    cache_ttl_seconds: int = Field(default=3600, ge=60)

    # ── API ───────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: list[str] = ["*"]

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info: object) -> int:  # type: ignore[override]
        # Validated after chunk_size is parsed
        return v

    @model_validator(mode="after")
    def validate_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.top_k_final > self.top_k_retrieval:
            raise ValueError(
                f"top_k_final ({self.top_k_final}) cannot exceed "
                f"top_k_retrieval ({self.top_k_retrieval})"
            )
        if self.llm_provider in ("openai", "anthropic") and not self.llm_api_key:
            raise ValueError(
                f"LLM_API_KEY is required when LLM_PROVIDER='{self.llm_provider}'"
            )
        return self

    def collection_path(self, name: str) -> Path:
        """Return the directory that stores a named collection's indexes."""
        return self.index_dir / name


# Module-level singleton — imported everywhere as `from app.core.config import settings`
settings = Settings()
