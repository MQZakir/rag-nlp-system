# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed for some native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 raguser \
    && useradd --uid 1001 --gid raguser --shell /bin/bash --create-home raguser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=raguser:raguser app/ ./app/
COPY --chown=raguser:raguser scripts/ ./scripts/

# Create data directories with correct permissions
RUN mkdir -p data/raw data/processed \
    && chown -R raguser:raguser data/

# HuggingFace model cache — pre-populate at build time or mount at runtime
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chown -R raguser:raguser /app/.cache

USER raguser

# Expose API port
EXPOSE 8000

# Health check — allows Docker / Compose to know when the API is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
