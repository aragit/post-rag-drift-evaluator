<h1 align="center">Post-RAG Drift Evaluator</h1>
<p align="center"><b>Naive RAG vs. Agentic RAG Benchmark — Embedding Drift, Faithfulness, Context Precision</b></p>

<p align="center"><sub>Python 3.11 · litellm · pgvector · Polars · scikit-learn · scipy · Docker · pytest · ruff · mypy · Streamlit</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" alt="Production Ready">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/pgvector-0.7+-blueviolet?logo=postgresql" alt="pgvector">
  <img src="https://img.shields.io/badge/litellm-1.35+-yellow" alt="litellm">
  <img src="https://img.shields.io/badge/Polars-1.0+-orange?logo=polars" alt="Polars">
  <img src="https://img.shields.io/badge/Docker-Multi--Stage-blue?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen?logo=githubactions" alt="CI">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

---

Framework-agnostic benchmark comparing Naive RAG against Agentic RAG across three critical axes: embedding distribution drift (Jensen-Shannon divergence), answer faithfulness, and context precision.

---

## Table of Contents

- [Why This Matters](#why-this-matters)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Local](#local)
  - [Docker](#docker)
  - [Dashboard](#dashboard)
- [Testing](#testing)
- [CI Pipeline](#ci-pipeline)
- [Configuration](#configuration)
- [Adding a Pipeline](#adding-a-pipeline)
- [Project Status](#project-status)

---

## Why This Matters

Retrieval-Augmented Generation (RAG) systems embed queries and documents into a shared latent space, then retrieve nearest neighbors to ground LLM generation. Over time, shifts in this embedding distribution -- caused by model updates, data distribution changes, or query pattern evolution -- degrade retrieval quality and introduce hallucination risk. This phenomenon, known as *latent space drift*, remains under-measured in production RAG deployments.

Agentic RAG attempts to mitigate these issues through query decomposition and multi-hop synthesis, but the literature lacks a standardized, quantitative framework to compare its robustness against simpler Naive (single-pass) RAG. This benchmark is differentiated from existing evaluation frameworks (RAGAS, TruLens, LangSmith) by the following design decisions:

- **Statistical drift detection over pointwise metrics.** Existing tools score individual retrievals but fail to measure population-level distribution shift. The DriftMonitor computes JS divergence between baseline and current embedding ensembles, providing an early-warning signal before per-query metrics degrade.

- **Real pgvector integration with parameterized cosine distance.** Rather than mocking or simulating retrieval, the pipelines execute live `<=>` vector search against a PostgreSQL/pgvector backend. This grounds benchmark results in realistic nearest-neighbor behavior, including index selectivity and distance distribution effects.

- **Multi-stage containerization with no build tools in runtime.** The Dockerfile separates dependency compilation (builder stage) from execution (runtime stage), producing an image that ships only `libpq5` -- no compilers, no headers, no attack surface beyond the application itself.

- **CI-enforced quality gates.** Every pull request passes through `ruff` linting, `ruff format --check`, `mypy` static type verification with `--ignore-missing-imports`, and `pytest` execution against a live pgvector service container. No untyped or unformatted code merges to main.

- **LLM-as-a-Judge with deterministic parse fallback.** Both `evaluate_faithfulness` and `evaluate_context_precision` demand `response_format="json_object"` and return `0.0` on any parse failure, eliminating silent scoring degradation from malformed LLM output.

---

## Architecture

```
evaluator/
  config.py              Pydantic settings loaded from .env with sensible defaults
  drift_monitor.py       PCA-to-1D reduction followed by Jensen-Shannon divergence
  benchmark.py           CLI harness: runs both pipelines, collects metrics, outputs Polars summary
  ui.py                  Streamlit dashboard for embedding drift telemetry
  rag_pipelines/
    base.py              RAGResponse pydantic model and BaseRAGPipeline abstract base class
    naive_rag.py         Single-embedding, flat top-k retrieval, single-turn generation
    agentic_rag.py       Query decomposition via LLM planner, per-sub-query retrieval, synthesis
  utils/
    metrics.py           LLM-as-Judge faithfulness and context precision scoring
scripts/
  seed_db.py             Creates pgvector extension, document_chunks table, seeds sample corpus
tests/
  test_drift_math.py     Unit tests for DriftMonitor under zero-drift and extreme-shift conditions
  test_e2e.py            Mocked end-to-end integration test for the benchmark harness
```

---

## Quick Start

### Local

```bash
pip install -r requirements.txt
python evaluator/benchmark.py
```

Run with custom queries:

```bash
python evaluator/benchmark.py --queries "What is the Q3 budget?" "Explain neural generation constraints"
```

### Docker

```bash
docker compose up -d --build
python scripts/seed_db.py
python evaluator/benchmark.py
```

### Dashboard

```bash
streamlit run evaluator/ui.py
```

---

## Testing

```bash
pytest tests/ -v --durations=10
```

The test suite covers:
- `test_drift_math.py` -- mathematical correctness of JS divergence using seeded numpy distributions, verifying near-zero drift for identical inputs and threshold-exceeding drift for shifted distributions
- `test_e2e.py` -- mocked integration test validating that `run_benchmark` produces a correctly shaped Polars DataFrame with both pipeline rows

---

## CI Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) enforces the following order on every push and pull request to `main`:

1. `ruff check evaluator/` -- linting
2. `ruff format --check evaluator/` -- formatting verification
3. `mypy --ignore-missing-imports evaluator/` -- static type checking
4. `pytest tests/ -v --durations=10` -- test execution

The pipeline provisions a `pgvector/pgvector:pg16` service container with health checks for integration test compatibility.

---

## Configuration

Configuration is managed through `EvaluatorConfig` in `evaluator/config.py`, which reads from a `.env` file or falls back to defaults:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/rag_db` | PostgreSQL connection string |
| `LITELLM_MASTER_KEY` | `sk-mock-key-1234` | API key for litellm routing |
| `DEFAULT_MODEL` | `gemma-3n-it` | Model for generation and LLM-as-Judge |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for embedding generation |

---

## Adding a Pipeline

Subclass `BaseRAGPipeline` in a new file under `evaluator/rag_pipelines/`, implement `execute(self, query: str) -> RAGResponse`, and register the class in `benchmark.py`'s `pipelines` dictionary.

---

## Project Status

Production-ready. Fully containerized, CI-protected, and instrumented with quantitative metrics for both retrieval quality and embedding distribution health.
