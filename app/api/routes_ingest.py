"""Ingest endpoint — add documents to a collection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.models.schemas import IngestRequest, IngestResponse
from app.services.chunker import RecursiveChunker
from app.services.pipeline import get_metrics
from app.services.retriever import get_retriever
from app.core.config import settings

log = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest documents into a collection",
    description=(
        "Chunk, embed, and index the provided texts. "
        "If the collection does not exist it will be created. "
        "The index is persisted to disk after each ingestion."
    ),
)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    if request.metadatas and len(request.metadatas) != len(request.texts):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="metadatas length must match texts length",
        )

    log.info(
        "ingestion started",
        collection=request.collection,
        documents=len(request.texts),
    )

    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    try:
        chunks = chunker.split_batch(
            texts=request.texts,
            metadatas=request.metadatas,
        )
    except Exception as exc:
        log.exception("chunking failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking error: {exc}",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No non-empty chunks produced from the provided texts.",
        )

    retriever = get_retriever()

    try:
        retriever.add_chunks(collection=request.collection, chunks=chunks)
        retriever.save_collection(request.collection)
    except Exception as exc:
        log.exception("indexing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing error: {exc}",
        ) from exc

    get_metrics().total_ingestions += 1

    log.info(
        "ingestion complete",
        collection=request.collection,
        documents=len(request.texts),
        chunks=len(chunks),
    )

    return IngestResponse(
        collection=request.collection,
        documents_ingested=len(request.texts),
        chunks_created=len(chunks),
        message=f"Successfully ingested {len(request.texts)} document(s) into '{request.collection}'.",
    )
