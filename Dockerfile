# Multi-stage Dockerfile for sentrix-evaluator
# Stage 1: Builder — install the package + dependencies into an isolated venv
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build dependencies required when compiling native extensions (no-op for
# pure-wheel deps, but required by polars/numpy/scipy fallback builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
        libffi-dev \
        make \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel setuptools

WORKDIR /build
COPY pyproject.toml /build/
COPY evaluator/ /build/evaluator/
COPY api/ /build/api/
COPY cli/ /build/cli/
COPY ingestion/ /build/ingestion/
COPY alerting/ /build/alerting/
COPY scripts/ /build/scripts/

# Install the application (builds the sdist and installs wheels into the venv)
RUN pip install .

# Stage 2: Runtime — minimal, non-root image, metrics served by the app on :8000
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user (uid/gid 10001) — the app never needs root privileges
RUN groupadd --gid 10001 sentrix && \
    useradd --uid 10001 --gid sentrix --create-home --shell /bin/sh sentrix

WORKDIR /app
COPY --chown=sentrix:sentrix pyproject.toml /app/

# Prometheus metrics are exposed via GET /metrics on port 8000 (no separate port)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:${PORT}/health', timeout=2).status == 200 else 1)"

USER sentrix

# Production entrypoint. `api.app:app` is the FastAPI application object
# (module-level `app = create_app()`); `sentrix-serve` console script is
# installed but uvicorn is invoked directly for deterministic multi-worker boot.
ENTRYPOINT ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
