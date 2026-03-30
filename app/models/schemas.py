"""
Request and response schemas.

Pydantic v2 models are used throughout — FastAPI uses these for
request validation, serialisation, and OpenAPI spec generation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Ingest ────────────────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """Ingest one or more text documents into a named collection."""

    texts: list[str] = Field(..., min_length=1, description="Raw document texts to ingest")
    metadatas: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional metadata dicts, one per document. If provided, must be the same length as texts.",
    )
    collection: str = Field(
        default="default",
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Collection name (alphanumeric, underscores, hyphens).",
    )

    @field_validator("metadatas")
    @classmethod
    def metadatas_match_texts(
        cls, v: list[dict[str, Any]] | None, info: object  # type: ignore[override]
    ) -> list[dict[str, Any]] | None:
        return v


class IngestResponse(BaseModel):
    collection: str
    documents_ingested: int
    chunks_created: int
    message: str


# ── Query ─────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Query a collection using hybrid retrieval + optional reranking."""

    query: str = Field(..., min_length=1, max_length=2048)
    collection: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]+$")
    top_k: int = Field(default=5, ge=1, le=50)
    use_reranker: bool = Field(default=True)
    stream: bool = Field(default=False)


class SourceDocument(BaseModel):
    content: str
    metadata: dict[str, Any]
    score: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    retrieval_method: str
    collection: str
    latency_ms: int


# ── Collections ───────────────────────────────────────────────────────────────


class CollectionInfo(BaseModel):
    name: str
    document_count: int
    chunk_count: int
    index_size_mb: float


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]
    total: int


class DeleteResponse(BaseModel):
    collection: str
    deleted: bool
    message: str


# ── Health / metrics ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    collections_loaded: int
    uptime_seconds: int
    embedding_model: str
    llm_provider: str
    llm_model: str


class MetricsResponse(BaseModel):
    total_queries: int
    total_ingestions: int
    cache_hits: int
    cache_misses: int
    avg_latency_ms: float
    collections: list[CollectionInfo]


# ── Error ─────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    code: int
