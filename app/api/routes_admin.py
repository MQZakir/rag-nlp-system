"""Admin endpoints — collection management, health, metrics."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    CollectionInfo,
    CollectionsResponse,
    DeleteResponse,
    HealthResponse,
    MetricsResponse,
)
from app.services.pipeline import get_metrics
from app.services.retriever import get_retriever

log = get_logger(__name__)
router = APIRouter(tags=["Admin"])

_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health() -> HealthResponse:
    retriever = get_retriever()
    return HealthResponse(
        status="ok",
        collections_loaded=len(retriever.collection_names()),
        uptime_seconds=int(time.time() - _start_time),
        embedding_model=settings.embedding_model,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Runtime metrics",
)
async def metrics() -> MetricsResponse:
    retriever = get_retriever()
    m = get_metrics()

    collection_infos: list[CollectionInfo] = []
    for name in retriever.collection_names():
        info = retriever.collection_info(name)
        collection_infos.append(
            CollectionInfo(
                name=info["name"],
                document_count=info["document_count"],
                chunk_count=info["chunk_count"],
                index_size_mb=info["index_size_mb"],
            )
        )

    return MetricsResponse(
        total_queries=m.total_queries,
        total_ingestions=m.total_ingestions,
        cache_hits=m.cache_hits,
        cache_misses=m.cache_misses,
        avg_latency_ms=m.avg_latency_ms,
        collections=collection_infos,
    )


@router.get(
    "/collections",
    response_model=CollectionsResponse,
    summary="List all collections",
)
async def list_collections() -> CollectionsResponse:
    retriever = get_retriever()
    infos: list[CollectionInfo] = []

    for name in retriever.collection_names():
        raw = retriever.collection_info(name)
        infos.append(
            CollectionInfo(
                name=raw["name"],
                document_count=raw["document_count"],
                chunk_count=raw["chunk_count"],
                index_size_mb=raw["index_size_mb"],
            )
        )

    return CollectionsResponse(collections=infos, total=len(infos))


@router.delete(
    "/collections/{collection_name}",
    response_model=DeleteResponse,
    summary="Delete a collection",
)
async def delete_collection(collection_name: str) -> DeleteResponse:
    retriever = get_retriever()

    if not retriever.has_collection(collection_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found.",
        )

    try:
        retriever.delete_collection(collection_name)
    except Exception as exc:
        log.exception("delete failed", collection=collection_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return DeleteResponse(
        collection=collection_name,
        deleted=True,
        message=f"Collection '{collection_name}' deleted successfully.",
    )
