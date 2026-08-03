# Sentrix Evaluator

### Causal Evaluation & Autonomous Optimization Engine for AI Systems

**Sentrix Evaluator** is a distribution-aware evaluation framework that detects latent-space drift, explains its root causes, simulates counterfactual alternatives, and recommends concrete remediation actions — transforming evaluation from passive monitoring into active decision-making.

It treats AI systems as **dynamic statistical processes**, enabling early detection of systemic failure modes before they manifest in degraded outputs.

---

## Why Sentrix Evaluator?

Modern AI systems operate under an assumption:

> The distribution of incoming data matches the distribution of baseline/reference data.

This assumption **breaks in production**.

When it does:
- System behavior becomes unpredictable
- Output quality degrades silently
- Failure modes cluster in unseen regions

Traditional evaluation tools measure:

```text
Input → Output correctness
```

This is **pointwise evaluation**.

Sentrix Evaluator measures **distribution-level alignment**:

```text
Distribution(Past) ≠ Distribution(Current)
```

Even if individual outputs appear correct, **systematic drift** accumulates — and systems fail **statistically, not individually**.

---

## Theoretical Foundations

### Latent Space Drift

AI systems operate over **vector spaces** that encode semantic meaning. Over time, system changes distort this space:

- **Model Updates** — Changing system configurations or pipeline versions shifts vector geometry
- **Data Distribution Shift** — New query patterns alter the density and topology of evaluated data
- **Configuration Drift** — Parameter changes (thresholds, model settings) incrementally degrade performance

These forces create **distribution misalignment** between:
- Historical baseline (reference state)
- Current state (latest evaluations)

### Why Distribution-Level Evaluation Matters

Pointwise metrics can mask systemic degradation. A system may produce correct outputs on most queries while **silently failing on emerging clusters** of inputs it has never seen before.

Sentrix detects this through:

1. **Jensen-Shannon Divergence (JSD)**
   Measures divergence between baseline and current metric distributions:

   ```text
   0 ≤ JSD(P || Q) ≤ 1
   ```
   - `0` → identical distributions
   - `> threshold` → significant drift requiring intervention

2. **Causal Attribution**
   Once drift is detected, heuristic scoring ranks system changes by their contribution to the observed shift — answering *"which change caused this?"*

3. **Counterfactual Simulation**
   Each ranked change is simulated in reverse: *"What if this change had not occurred?"* The engine clones the history store, reverts the change, and re-estimates metric values deterministically.

4. **Optimization Engine**
   The most impactful counterfactual is translated into a concrete, ranked recommendation: *"Revert model in run_42 (expected improvement: 0.40, confidence: 92%)."*

This end-to-end pipeline converts **statistical signals** into **actionable decisions**.

---

## Core Capabilities

| Capability | What it does |
|---|---|
| **Drift Detection** | Sliding-window mean-shift detection on any metric series using Jensen-Shannon divergence |
| **Causal Attribution** | Heuristic scoring ranks system changes by contribution to detected drift |
| **Counterfactual Simulation** | Simulates "what if this change had not occurred?" via non-mutating store cloning |
| **Optimization Engine** | Generates ranked, actionable remediation recommendations from simulation results |
| **API Layer** | RESTful HTTP endpoints exposing the full pipeline (/drift, /attribution, /counterfactual, /optimize) |
| **CLI Interface** | Diagnostic commands for store inspection and drift evaluation |

---

## Pipeline Overview

```
JSONHistoryStore
      │
      ▼
   DriftEvent           (Phase 4 — temporal drift detection)
      │
      ▼
CausalAttribution       (Phase 5 — root-cause ranking)
      │
      ▼
CounterfactualResult    (Phase 6 — simulation)
      │
      ▼
OptimizationPlan        (Phase 7 — actionable recommendations)
      │
      ▼
      API                 (Phase 8 — HTTP endpoints)
```

Each stage consumes the output of the previous one. The pipeline is **fully deterministic** — no randomness, no ML libraries, no hidden state.

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
├── storage/            # File-backed JSONL history persistence
│   ├── json_store.py    # JSONHistoryStore
│   └── models.py        # EvaluationRecord
└── metrics/            # Metric definitions
    ├── drift/           # Drift metrics (Jensen-Shannon divergence)
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

cli/
└── drift_cli.py        # CLI diagnostic tool

scripts/
└── seed_db.py          # Database initialization

tests/
├── test_api.py                   # Phase 8: API integration tests
├── test_counterfactual.py        # Phase 6: simulation tests
├── test_optimization.py          # Phase 7: optimization tests
├── test_causal_attribution.py    # Phase 5: attribution tests
├── test_temporal_analysis.py     # Phase 4: drift detection tests
├── test_history_store.py         # Phase 3: storage tests
├── test_drift_math.py            # Drift math correctness
├── test_drift_properties.py      # Distribution property tests
└── test_run_schema.py            # Phase 1: schema validation
```

### Module Summary

| Module | Responsibility |
|---|---|
| `evaluator/temporal/` | Detects drift events from metric time-series using sliding-window mean-shift with JSD |
| `evaluator/causal/` | Extracts system-change events, builds feature vectors, scores causal impact via heuristic weighted composite |
| `evaluator/counterfactual/` | Simulates interventions by cloning the history store and reverting changes non-mutatingly |
| `evaluator/optimization/` | Maps changes to remediation actions, scores expected improvement, ranks recommendations |
| `evaluator/storage/` | JSONL-backed history store for evaluation records |
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

if events:
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
python -m cli.drift_cli evaluate --baseline-id group_a --current-id group_b
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
  "summary": "Top recommendation: revert model change in run run_42 (expected improvement: 0.4)",
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

1. **Distribution-First Evaluation** — Measures systemic drift, not just pointwise correctness
2. **Deterministic Reproducibility** — Every operation produces identical output given identical input. No randomness, no ML dependencies.
3. **Non-Mutating Simulation** — Counterfactual analysis clones the history store; original data is never modified
4. **Modular Architecture** — Each phase (detect, explain, simulate, act) is a standalone module with clear interfaces
5. **Actionable Insights** — Converts statistical signals into ranked, concrete recommendations
6. **API-Core Separation** — The FastAPI layer wraps core logic with Pydantic schemas; internal dataclasses are never exposed directly
7. **Stable Cross-Module Identity** — `change_id` values use deterministic UUIDv5 for stable references across all pipeline stages

---

## Testing

```bash
pytest
```

```bash
ruff check
```

The test suite covers all phases (1–8) including:
- Schema validation and serialization round-trips
- Drift detection math correctness
- Causal attribution scoring and ranking
- Counterfactual simulation (store cloning, metric re-estimation)
- Optimization action generation and recommendation ranking
- API endpoint behavior (request/response, error handling, empty stores)
- End-to-end pipeline integration

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
| **ML-Augmented Attribution** | Research |

---

## Contributing

Pull requests, experiments, and research extensions are welcome.

---

## License

MIT

Copyright (c) 2026 Arash Nicoomanesh (aragit)
