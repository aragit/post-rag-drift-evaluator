# Multi-stage Dockerfile for sentrix-evaluator
# Stage 1: Builder — compile wheels and create virtualenv
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    musl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv with build context
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Copy project files and build
WORKDIR /build
COPY pyproject.toml /build/

# Install build tools
RUN pip install build

# Copy source code
COPY evaluator/ /build/evaluator/
COPY api/ /build/api/
COPY cli/ /build/cli/
COPY ingestion/ /build/ingestion/
COPY alerting/ /build/alerting/
COPY scripts/ /build/scripts/

# Install the application in the venv (this builds and installs the package)
RUN pip install .

# Stage 2: Runtime — minimal image with non-root user
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user and group
RUN groupadd --gid 10001 sentrix && \
    useradd --uid 10001 --gid sentrix --create-home --shell /bin/sh sentrix

# Create app directory
RUN mkdir -p /app && chown sentrix:sentrix /app
WORKDIR /app

# Copy additional project files needed at runtime
COPY --chown=sentrix:sentrix pyproject.toml /app/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/v1/eval')" || exit 1

# Expose port
EXPOSE 8000

# Run as non-root user
USER sentrix

# Entrypoint
ENTRYPOINT ["uvicorn", "evaluator.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
