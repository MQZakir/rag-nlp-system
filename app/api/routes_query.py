"""Query endpoints — standard and streaming."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.core.logging import get_logger
from app.models.schemas import QueryRequest, QueryResponse
from app.services.pipeline import get_pipeline
from app.services.retriever import get_retriever

log = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query a collection (blocking)",
    description=(
        "Retrieve relevant chunks via hybrid search, optionally rerank with a "
        "cross-encoder, and generate a grounded answer. "
        "For streaming responses use GET /query/stream."
    ),
)
async def query(request: QueryRequest) -> QueryResponse:
    retriever = get_retriever()

    if not retriever.has_collection(request.collection):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{request.collection}' not found. Ingest documents first.",
        )

    try:
        return await get_pipeline().query(
            query=request.query,
            collection=request.collection,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )
    except Exception as exc:
        log.exception("query failed", collection=request.collection)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get(
    "/stream",
    summary="Query a collection (SSE streaming)",
    description=(
        "Server-sent events stream. Tokens are emitted as `data: <token>\\n\\n` events. "
        "The stream closes with `data: [DONE]\\n\\n`."
    ),
    response_class=StreamingResponse,
)
async def query_stream(
    query: str = Query(..., min_length=1, max_length=2048, description="The query string"),
    collection: str = Query(default="default", pattern=r"^[a-zA-Z0-9_-]+$"),
    top_k: int = Query(default=5, ge=1, le=50),
    use_reranker: bool = Query(default=True),
) -> StreamingResponse:
    retriever = get_retriever()

    if not retriever.has_collection(collection):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection}' not found.",
        )

    async def event_generator():
        try:
            async for token in get_pipeline().stream(
                query=query,
                collection=collection,
                top_k=top_k,
                use_reranker=use_reranker,
            ):
                # SSE format: each event is "data: <payload>\n\n"
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            log.exception("stream error", collection=collection)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering for SSE
        },
    )
