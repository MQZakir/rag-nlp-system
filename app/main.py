"""
FastAPI application factory.

The ``lifespan`` context manager handles startup and shutdown:
  - Startup: configure logging, create index directory, load persisted collections
  - Shutdown: nothing to teardown (FAISS indexes live in memory; saves happen at ingest)

Middleware:
  - CORS (configurable via CORS_ORIGINS env var)
  - Request ID injection (X-Request-ID header)
  - Logging of every request with method, path, status, and latency
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_admin import router as admin_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.models.schemas import ErrorResponse
from app.services.retriever import get_retriever

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── Startup ───────────────────────────────────────────────────────────────
    configure_logging()
    log.info(
        "starting RAG system",
        embedding_model=settings.embedding_model,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )

    settings.index_dir.mkdir(parents=True, exist_ok=True)

    retriever = get_retriever()
    retriever.load_all()
    log.info("collections loaded", collections=retriever.collection_names())

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG-Powered NLP System",
        description=(
            "Retrieval-augmented generation pipeline with hybrid semantic search "
            "(FAISS + BM25), cross-encoder reranking, and LangChain-powered generation."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled exception", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc),
                code=500,
            ).model_dump(),
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(admin_router)

    return app


app = create_app()
