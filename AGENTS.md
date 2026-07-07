# post-rag-drift-evaluator

Framework-agnostic benchmark comparing Naive RAG vs. Agentic RAG on embedding drift (JS-divergence), faithfulness, and context precision.

## Architecture

```
evaluator/
  config.py           — Pydantic settings via .env
  drift_monitor.py    — DriftMonitor: PCA→1D + Jensen-Shannon divergence
  benchmark.py        — CLI harness, runs both pipelines, outputs Polars summary
  rag_pipelines/
    base.py           — RAGResponse pydantic model + BaseRAGPipeline ABC
    naive_rag.py      — Flat retrieval + single-turn prompt
    agentic_rag.py    — Query decomposition → multi-hop synthesis
  utils/
    metrics.py        — LLM-as-Judge faithfulness & context precision (litellm)
tests/
  test_drift_math.py  — DriftMonitor unit tests
```

## Key Commands

```bash
pip install -r requirements.txt
python evaluator/benchmark.py                     # runs with 2 default queries
python evaluator/benchmark.py --queries "Q1" "Q2"
pytest tests/ -v --durations=10                   # run unit tests
ruff check evaluator/ && ruff format --check evaluator/  # lint & format
mypy --ignore-missing-imports evaluator/          # static types
docker compose up --build                         # full stack with pgvector
```

## Important Details

- **DriftMonitor** reduces embeddings to 1D via PCA before computing JS divergence. Threshold defaults to 0.15.
- **LLM-as-Judge** metrics call configured `DEFAULT_MODEL` via litellm with `response_format="json_object"`. Both return `0.0` on parse failure.
- **Pipelines use real pgvector** via psycopg2 parameterized queries (`<=>` cosine distance on `document_chunks` table). DB connectivity failure degrades to static fallback.
- **Docker** multi-stage build — builder stage compiles `libpq-dev` deps, runtime stage ships only `libpq5`.
- **Config** reads from `.env` or defaults (mock key `sk-mock-key-1234`, localhost PG, `gemma-3n-it` model).
- Added pipelines register in `benchmark.py`'s `pipelines` dict.

## CI Pipeline (`.github/workflows/ci.yml`)

Order: `ruff check` → `ruff format --check` → `mypy` → `pytest`. Requires pgvector service container.

## Testing

- `tests/test_drift_math.py` uses seeded numpy to verify zero-drift (≈0.0) and extreme-shift (>threshold) math.
- Add tests under `tests/` mirroring `evaluator/` layout. Use `pytest`.
