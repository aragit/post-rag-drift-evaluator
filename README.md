# Sentrix Evaluator

**Causal Evaluation & Autonomous Optimization Engine for AI Systems**

---

## Badges

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com/aragit/post-rag-drift-evaluator)

---

## Introduction

Sentrix Evaluator detects drift, explains its root causes, simulates counterfactual alternatives, and recommends concrete actions to fix problems — all from a single evaluation history store.

Built for AI teams that need to move beyond passive monitoring to **reasoning about why drift occurred** and **what to do about it**.

**Measure → Track → Detect → Explain → Simulate → Act**

---

## Why Sentrix Evaluator?

Traditional monitoring tools report *that* something changed:

| Traditional Tools | Sentrix Evaluator |
|---|---|
| Metrics dashboards | Causal attribution |
| Threshold alerts | Root-cause ranking |
| Manual investigation | Counterfactual simulation |
| Reactive response | Autonomous recommendations |

Sentrix doesn't just tell you drift happened — it tells you **which change caused it**, **what would happen if you removed it**, and **what action to take**.

---

## Core Capabilities

| Capability | Description |
|---|---|
| **Drift Detection** | Sliding-window mean-shift detection over any metric series |
| **Causal Attribution** | Heuristic scoring ranks system changes by contribution to drift |
| **Counterfactual Simulation** | Simulates "what if this change had not occurred?" via store cloning & metric re-estimation |
| **Optimization Engine** | Converts causal factors + counterfactuals into ranked, actionable recommendations |
| **API Layer** | RESTful HTTP endpoints for all pipeline stages |
| **CLI** | Diagnostic commands for store inspection and drift evaluation |

---

## Pipeline Overview

```
JSONHistoryStore
      │
      ▼
   DriftEvent           (Phase 4 — temporal drift detection)
      │
      ▼
CausalAttribution        (Phase 5 — root-cause ranking)
      │
      ▼
CounterfactualResult     (Phase 6 — simulation)
      │
      ▼
OptimizationPlan        (Phase 7 — actionable recommendations)
      │
      ▼
      API                 (Phase 8 — HTTP endpoints)
```

Each stage consumes the output of the previous one. The pipeline is fully deterministic — no randomness, no ML dependencies, no hidden state.

---

## Architecture

```
evaluator/
├── temporal/           # Drift event detection & time-series analysis
│   ├── drift_detection.py
│   ├── models.py          # DriftEvent
│   └── series.py
├── causal/             # Causal attribution
│   ├── attribution.py
│   ├── change_extractor.py
│   ├── feature_builder.py
│   └── models.py          # ChangeEvent, CausalFactor, CausalAttribution
├── counterfactual/     # Counterfactual simulation
│   ├── simulator.py
│   ├── scenario.py
│   ├── estimator.py
│   └── models.py          # Intervention, CounterfactualScenario, CounterfactualResult
├── optimization/       # Optimization & recommendations
│   ├── optimizer.py
│   ├── actions.py
│   ├── scorer.py
│   └── models.py          # OptimizationAction, OptimizationRecommendation, OptimizationPlan
├── storage/            # History persistence
│   ├── json_store.py    # JSONHistoryStore
│   └── models.py        # EvaluationRecord
└── metrics/            # Metric definitions
    ├── drift/           # Drift metrics (e.g., Jensen-Shannon divergence)
    ├── quality/         # Quality metrics
    └── results.py       # MetricResult, DriftResult, QualityResult

api/
├── routes/             # FastAPI endpoint handlers
│   ├── drift.py
│   ├── attribution.py
│   ├── counterfactual.py
│   └── optimization.py
├── dependencies.py     # Dependency injection (get_store)
├── schemas.py          # Pydantic request/response models
├── main.py             # FastAPI app factory
└── config.py           # Application configuration

cli/                    # CLI diagnostic tool
└── drift_cli.py
```

### Module Summary

| Module | Responsibility |
|---|---|
| `evaluator/temporal/` | Detects drift events from metric series using sliding-window mean-shift |
| `evaluator/causal/` | Extracts system-change events, builds feature vectors, scores causal impact |
| `evaluator/counterfactual/` | Simulates interventions by cloning the history store and reverting changes |
| `evaluator/optimization/` | Maps changes to concrete actions, scores expected improvement, ranks recommendations |
| `evaluator/storage/` | File-backed JSONL history store for evaluation records |
| `evaluator/metrics/` | Metric computation (drift + quality) with structured results |
| `api/` | HTTP API exposing the full pipeline as REST endpoints |
| `cli/` | Command-line diagnostics and alerting |

---

## Installation

```bash
pip install sentrix-evaluator
```

For development:

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### Python Usage

```python
from evaluator.storage import JSONHistoryStore
from evaluator.temporal.drift_detection import detect_drift_from_store
from evaluator.causal.attribution import attribute_drift
from evaluator.counterfactual.simulator import run_counterfactual_analysis
from evaluator.optimization.optimizer import generate_optimization_plan

# Load evaluation history
store = JSONHistoryStore("history.jsonl")

# 1. Detect drift
events = detect_drift_from_store(
    store, metric_name="js_divergence", window_size=3, threshold=0.15
)

# 2. Explain root cause
attribution = attribute_drift(events[0], store)

# 3. Simulate alternatives
counterfx = run_counterfactual_analysis(events[0], attribution, store)

# 4. Get recommendations
plan = generate_optimization_plan(events[0], attribution, counterfx)
print(plan.summary)
```

### API Usage

```bash
uvicorn api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"metric_name": "js_divergence", "window_size": 3, "threshold": 0.15}'
```

### CLI Usage

```bash
python -m cli.drift_cli stats
python -m cli.drift_cli evaluate --baseline-id naive --current-id agentic
```

---

## API Endpoints

All endpoints use `POST` and accept JSON request bodies. The history store path is configurable via the `EVALUATOR_HISTORY_PATH` environment variable or the `store_path` argument to `create_app()`.

| Endpoint | Description |
|---|---|
| `POST /drift` | Detect drift events from the history store. Accepts `metric_name`, `window_size`, `threshold`. |
| `POST /attribution` | Run causal attribution on a drift event. Accepts a serialized `DriftEvent`. |
| `POST /counterfactual` | Run counterfactual simulation. Accepts a `DriftEvent` and `CausalAttribution`. |
| `POST /optimize` | Full end-to-end pipeline: detect → attribute → simulate → recommend. Accepts drift detection parameters. |

---

## Example Output

### Optimization Plan

```json
{
  "plan_id": "a1b2c3d4-...",
  "drift_event_id": "ev-7f3e...",
  "recommendations": [
    {
      "recommendation_id": "rec-abc...",
      "action": {
        "action_id": "act-def...",
        "action_type": "revert_model",
        "target_run_id": "run_42",
        "change_id": "chg-123...",
        "description": "Revert model change",
        "metadata": {
          "factor_name": "model_update",
          "factor_score": 0.92
        }
      },
      "expected_improvement": 0.40,
      "confidence": 0.92,
      "priority": 1,
      "metadata": {
        "source": "counterfactual_simulation"
      }
    }
  ],
  "summary": "Top recommendation: revert model in run run_42 (expected improvement: 0.4)",
  "metadata": {
    "metric_name": "js_divergence",
    "num_actions": 1,
    "num_recommendations": 1,
    "drift_magnitude": 0.45
  }
}
```

---

## Design Principles

1. **Deterministic** — Every operation produces identical output given identical input. No randomness, no ML libraries.
2. **Modular Architecture** — Each phase (detect, explain, simulate, act) is a standalone module with clear interfaces.
3. **No ML Dependency (Yet)** — Causal scoring uses deterministic heuristic weights, not trained models.
4. **Reproducibility** — `change_id` values are generated as deterministic UUIDv5, ensuring stable cross-module references across runs.
5. **Non-Mutating** — Counterfactual simulation clones the history store; the original data is never modified.
6. **API-Core Separation** — The FastAPI layer wraps core logic with Pydantic schemas; internal dataclasses are never exposed directly.

---

## Testing

```bash
pytest
```

```bash
ruff check
```

Test suite covers all phases (4–8) including model serialization, pipeline integration, endpoint behavior, and edge cases (empty stores, invalid payloads, no-drift scenarios).

---

## Roadmap

| Feature | Status |
|---|---|
| Drift Detection | Complete |
| Causal Attribution | Complete |
| Counterfactual Simulation | Complete |
| Optimization Engine | Complete |
| API Layer | Complete |
| **Dashboard UI** | Planned |
| **Autonomous Execution Loop** | Planned |
| **CI/CD Integration** | Planned |
| **ML-augmented Attribution** | Research |

---

## License

MIT

Copyright (c) 2026 Arash Nicoomanesh (aragit)
