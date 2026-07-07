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
```

## Key Commands

```bash
pip install -r requirements.txt
python evaluator/benchmark.py    # runs with 2 default queries
python evaluator/benchmark.py --queries "Q1" "Q2" "..."
```

## Important Details

- **DriftMonitor** reduces embeddings to 1D via PCA before computing JS divergence. Threshold defaults to 0.15.
- **LLM-as-Judge** metrics call the configured `DEFAULT_MODEL` via litellm with `response_format="json_object"`. Both return `0.0` on parse failure.
- **Pipelines use mock retrieval** — no real pgvector or database connection needed. Drop-in real retrieval by replacing `_mock_pgvector_retrieval`.
- **Config** reads from `.env` or defaults (mock key `sk-mock-key-1234`, localhost PG, `gemma-3n-it` model).
- Added pipelines register in `benchmark.py`'s `pipelines` dict.

## Testing

No tests yet. Add tests under `tests/` mirroring the `evaluator/` layout. Use `pytest`.
