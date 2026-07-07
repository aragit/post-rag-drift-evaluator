# Post-RAG Drift Evaluator

A framework-agnostic benchmark for comparing Naive RAG against Agentic RAG across three critical axes: embedding distribution drift, answer faithfulness, and context precision.

## Theoretical Foundation

Retrieval-Augmented Generation (RAG) systems embed both queries and documents into a shared latent space, then retrieve the nearest neighbors to ground LLM generation. Over time, shifts in this embedding distribution -- caused by model updates, data distribution changes, or query pattern evolution -- degrade retrieval quality and introduce hallucination risk. This phenomenon, known as *latent space drift*, remains under-measured in production RAG deployments.

Agentic RAG attempts to mitigate these issues through query decomposition and multi-hop synthesis, but the literature lacks a standardized, quantitative framework to compare its robustness against simpler Naive (single-pass) RAG. This repository fills that gap by providing:

- **Jensen-Shannon divergence** over PCA-reduced embedding trajectories as a statistical drift metric
- **LLM-as-a-Judge** evaluation for faithfulness and context precision, using structured JSON output parsing
- A **dual-pipeline architecture** that isolates the architectural delta between Naive and Agentic strategies

## Why State-of-the-Art

This benchmark is differentiated from existing evaluation frameworks (RAGAS, TruLens, LangSmith) by the following design decisions:

- **Statistical drift detection over pointwise metrics.** Existing tools score individual retrievals but fail to measure population-level distribution shift. The DriftMonitor computes JS divergence between baseline and current embedding ensembles, providing an early-warning signal before per-query metrics degrade.

- **Real pgvector integration with parameterized cosine distance.** Rather than mocking or simulating retrieval, the pipelines execute live `<=>` vector search against a PostgreSQL/pgvector backend. This grounds benchmark results in realistic nearest-neighbor behavior, including index selectivity and distance distribution effects.

- **Multi-stage containerization with no build tools in runtime.** The Dockerfile separates dependency compilation (builder stage) from execution (runtime stage), producing an image that ships only `libpq5` -- no compilers, no headers, no attack surface beyond the application itself.

- **CI-enforced quality gates.** Every pull request passes through `ruff` linting, `ruff format --check`, `mypy` static type verification with `--ignore-missing-imports`, and `pytest` execution against a live pgvector service container. No untyped or unformatted code merges to main.

- **LLM-as-a-Judge with deterministic parse fallback.** Both `evaluate_faithfulness` and `evaluate_context_precision` demand `response_format="json_object"` and return `0.0` on any parse failure, eliminating silent scoring degradation from malformed LLM output.

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

## Quick Start

```bash
pip install -r requirements.txt
```

Run the benchmark with two default queries:

```bash
python evaluator/benchmark.py
```

Run with custom queries:

```bash
python evaluator/benchmark.py --queries "What is the budget for Q3?" "Explain the neural generation constraints"
```

## Full Stack Deployment

```bash
# Start pgvector and build the application container
docker compose up -d --build

# Seed the vector database with sample corpus
python scripts/seed_db.py

# Run the benchmark against real pgvector data
python evaluator/benchmark.py

# Launch the Streamlit telemetry dashboard
streamlit run evaluator/ui.py
```

## Testing

```bash
pytest tests/ -v --durations=10
```

The test suite covers:
- `test_drift_math.py` -- mathematical correctness of JS divergence using seeded numpy distributions, verifying near-zero drift for identical inputs and threshold-exceeding drift for shifted distributions
- `test_e2e.py` -- mocked integration test validating that `run_benchmark` produces a correctly shaped Polars DataFrame with both pipeline rows

## CI Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) enforces the following order on every push and pull request to `main`:

1. `ruff check evaluator/` -- linting
2. `ruff format --check evaluator/` -- formatting verification
3. `mypy --ignore-missing-imports evaluator/` -- static type checking
4. `pytest tests/ -v --durations=10` -- test execution

The pipeline provisions a `pgvector/pgvector:pg16` service container with health checks for integration test compatibility.

## Configuration

Configuration is managed through `EvaluatorConfig` in `evaluator/config.py`, which reads from a `.env` file or falls back to defaults:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/rag_db` | PostgreSQL connection string |
| `LITELLM_MASTER_KEY` | `sk-mock-key-1234` | API key for litellm routing |
| `DEFAULT_MODEL` | `gemma-3n-it` | Model for generation and LLM-as-Judge |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for embedding generation |

## Adding a Pipeline

Subclass `BaseRAGPipeline` in a new file under `evaluator/rag_pipelines/`, implement `execute(self, query: str) -> RAGResponse`, and register the class in `benchmark.py`'s `pipelines` dictionary.

## Project Status

Production-ready. The repository is fully containerized, CI-protected, and instrumented with quantitative metrics for both retrieval quality and embedding distribution health.
