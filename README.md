# Sentrix Evaluator (`post-rag-drift-evaluator`)

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/tests-201%20passed-success.svg)](#testing--verification)

**Sentrix Evaluator** is an enterprise-grade, zero-disk, closed-loop drift remediation engine for Retrieval-Augmented Generation (RAG) and LLM pipelines.

It bridges non-parametric latent embedding space monitoring with causal fault attribution, counterfactual impact simulation, track-aware optimization, and safety policy guardrails — delivering automated, real-time remediation without introducing disk I/O latency or unbounded operational feedback loops.

---

## Key Features

* **Dual-Track Latent Drift Detection:** Layer-separated divergence scoring across `"retrieval"` (vector representations) and `"generation"` (prompt/token contexts) tracks using Maximum Mean Discrepancy (MMD), Sliced Wasserstein Distance (SWD), and Jensen-Shannon Divergence (JSD).
* **Adaptive Threshold Learning:** Dynamic z-score sliding quantile bounds ($\mu_{\text{window}} + z \cdot \sigma_{\text{window}}$) with clamped absolute limits $[0.05, 0.50]$ that adaptively tune drift sensitivity based on ambient traffic noise.
* **Real-Time Streaming Drift Buffers:** Dual-track ring buffers with deterministic reservoir sampling and FIFO eviction strategies for continuous vector streaming without rigid batch boundaries.
* **Causal-Latent Fusion Layer:** Structural mapping connecting continuous embedding space divergence directly to Bayesian Causal DAG node failure priors ($P(\text{Node Failure}) = \min(1.0, \text{drift\_score} \times \text{sensitivity})$).
* **Zero-Disk Counterfactual Simulation:** In-memory state machine (`InMemoryHistoryStore`) running EWMA and OLS linear trend-adjusted counterfactual estimations without touching persistent disk.
* **Safety Guardrails & Closed-Loop Remediation:** Policy evaluation enforcing action cooldown periods (default 300s), anti-flapping throttles (max 5/hr), and hard scalar parameter bounds (`temperature` $\in [0, 1]$, `top_k` $\in [1, 50]$).
* **Production Gateway & Observability:** Production-ready FastAPI HTTP server, asynchronous streaming endpoints (`/v1/stream/*`), CLI suite (`sentrix`), and native OpenTelemetry/Prometheus metric exporters.

---

## System Architecture

```text
                                 +-----------------------------------+
                                 |  Continuous Vector Stream / API   |
                                 +-----------------------------------+
                                                   |
                                                   v
                                     [ StreamingDriftBuffer ]
                                     (Reservoir / FIFO Ring)
                                                   |
                                                   v
                                       [ LatentDriftEngine ]
                                 (Dual-Track MMD / SWD / JSD)
                                                   |
                                   +---------------+---------------+
                                   |                               |
                        [ Retrieval Track ]               [ Generation Track ]
                        (top_k, reranker)                 (temp, prompt, model)
                                   |                               |
                                   +---------------+---------------+
                                                   |
                                                   v
                                    [ AdaptiveThresholdManager ]
                                    (Dynamic Rolling Z-Score Bounds)
                                                   |
                                                   v
                                     [ CausalLatentFusionEngine ]
                                  (Updates Causal DAG Node Priors)
                                                   |
                                                   v
                                  [ Zero-Disk Simulation Engine ]
                                  (InMemoryHistoryStore + EWMA/OLS)
                                                   |
                                                   v
                                        [ PolicyEvaluator ]
                                   (Cooldown / Flapping / Bounds)
                                                   |
                                                   v
                                      [ OptimizationRunner ]
                          +------------------------+------------------------+
                          |                        |                        |
                 STATUS_APPROVED      STATUS_BLOCKED_BY_GUARDRAIL   STATUS_NO_ACTION_NEEDED
                          |                        |                        |
                          +------------------------+------------------------+
                                                   |
                                                   v
                                      [ SentrixMetricsExporter ]
                                    (OpenTelemetry / Prometheus)
```

---

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/post-rag-drift-evaluator.git
cd post-rag-drift-evaluator

# Install package with production dependencies
pip install -e .

# Or install with development and telemetry extras
pip install -e ".[dev,telemetry]"
```

### Docker Deployment

```bash
# Build local multi-stage container
docker build -t sentrix-evaluator:v0.6.1 .

# Run production stack (API + Postgres + Redis + Prometheus)
docker-compose -f docker-compose.prod.yml up -d
```

### Helm Chart (Kubernetes)

```bash
# Lint and render Helm chart
helm lint deploy/helm/sentrix-evaluator/
helm template sentrix-evaluator deploy/helm/sentrix-evaluator/

# Install to Kubernetes cluster
helm install sentrix-evaluator deploy/helm/sentrix-evaluator/ --namespace sentrix --create-namespace
```

---

## Python SDK Usage

### 1. Zero-Disk Closed-Loop Optimization

```python
import numpy as np
from evaluator.storage import InMemoryHistoryStore
from evaluator.guardrails import PolicyEvaluator
from evaluator.optimization import OptimizationEngine, OptimizationRunner
from evaluator.schema import DriftEvent

# Initialize in-memory store and engine components
history_store = InMemoryHistoryStore()
policy_evaluator = PolicyEvaluator(cooldown_seconds=300, max_flapping_per_hour=5)
optimization_engine = OptimizationEngine(min_confidence=0.70)
runner = OptimizationRunner(engine=optimization_engine, policy=policy_evaluator)

# Construct drift event on retrieval track
drift_event = DriftEvent(
    event_id="evt_001",
    track="retrieval",
    metric_name="retrieval_precision",
    current_value=0.52,
    baseline_value=0.85,
    severity="high",
)

# Run closed-loop optimization cycle
result = runner.run_optimization_cycle(
    drift_event=drift_event,
    history_store=history_store,
)

print(f"Optimization Status: {result.status}")
if result.status == "approved":
    print(f"Action Approved: {result.selected_action.action_type}")
    print(f"Parameters: {result.selected_action.parameters}")
```

### 2. Streaming Buffer & Dynamic Adaptive Drift Detection

```python
import numpy as np
from evaluator.latent_drift import (
    StreamingDriftBuffer,
    AdaptiveThresholdManager,
    LatentDriftEngine,
)

# Initialize 1000-sample streaming buffer and dynamic threshold manager
buffer = StreamingDriftBuffer(capacity=1000, sample_strategy="reservoir")
threshold_mgr = AdaptiveThresholdManager(base_threshold=0.15, sensitivity_z=2.0)
engine = LatentDriftEngine(threshold_manager=threshold_mgr)

# Ingest high-dimensional query embeddings from real-time stream
vectors = np.random.randn(100, 384)
for vec in vectors:
    buffer.ingest(vector=vec, track="retrieval")

# Flush current buffer snapshot and evaluate dynamic latent drift
batch = buffer.flush_batch()
if buffer.is_ready(min_samples=50):
    drift_result = engine.compute_drift(current_batch=batch, baseline_batch=batch)
    print(f"Drift Score: {drift_result.score:.4f}")
    print(f"Dynamic Threshold Used: {drift_result.metadata['dynamic_threshold']:.4f}")
```

### 3. Causal-Latent Fusion Layer

```python
from evaluator.causal import CausalGraph, CausalNode, CausalLatentFusionEngine
from evaluator.schema import LatentDriftResult

# Initialize Causal DAG
graph = CausalGraph()
graph.add_node(CausalNode(node_id="VectorIndexNode", node_type="retrieval", prior_failure_prob=0.05))
graph.add_node(CausalNode(node_id="LLMRouterNode", node_type="generation", prior_failure_prob=0.02))

# Simulate latent drift detection result
drift_result = LatentDriftResult(
    track="retrieval",
    score=0.42,
    metric_breakdown={"mmd": 0.38, "swd": 0.45},
    metadata={},
)

# Fuse latent drift signal directly into Causal DAG node priors
updated_graph = CausalLatentFusionEngine.fuse_drift_into_causal_graph(
    drift_result=drift_result,
    graph=graph,
)

retrieval_node = updated_graph.get_node("VectorIndexNode")
print(f"Updated Failure Probability: {retrieval_node.prior_failure_prob:.4f}")
```

---

## API Gateway & CLI Reference

### REST API Endpoints

| Endpoint | Method | Description | Payload / Query |
|---|---|---|---|
| `/v1/eval` | POST | Ingest raw EvaluationRecord batches into storage. | `{"records": [...]}` |
| `/v1/drift/detect` | POST | Triggers dual-track MMD/SWD detection across batches. | `{"current_batch_id": "...", "track": "retrieval"}` |
| `/v1/remediate` | POST | Executes full OptimizationRunner remediation cycle. | `{"drift_event": {...}}` |
| `/v1/stream/ingest` | POST | Ingests a single vector into the live streaming buffer. | `{"vector": [...], "track": "retrieval"}` |
| `/v1/stream/flush` | POST | Flushes stream buffer and evaluates current drift. | `{"track": "retrieval", "metric": "mmd"}` |

### CLI Subcommands (`sentrix`)

```bash
# Evaluate ingested evaluation records
sentrix eval --input records.json --store memory

# Trigger dual-track drift detection on vector batches
sentrix drift --current current_batch.json --baseline baseline_batch.json --track retrieval --metric mmd

# Stream line-delimited JSON vector payloads from stdin
cat stream_vectors.jsonl | sentrix stream --track generation --capacity 500 --flush

# Run closed-loop remediation on drift event
sentrix remediate --event drift_event.json --cooldown 300
```

---

## OpenTelemetry & Metrics

Sentrix native telemetry exposes metrics via standard OpenTelemetry meters with automatic fallback to no-op primitives when dependencies are omitted.

| Metric Name | Type | Labels / Dimensions | Description |
|---|---|---|---|
| `sentrix_latent_drift_score` | Gauge | `track`, `metric` | Current calculated latent distance score (MMD/SWD/JSD). |
| `sentrix_estimated_impact_delta` | Gauge | `metric_name` | Estimated impact delta from counterfactual simulations. |
| `sentrix_drift_events_total` | Counter | `severity` | Total count of detected drift events triggered. |
| `sentrix_optimization_actions_total` | Counter | `status`, `rule_violated` | Total optimization cycles categorized by approval status. |
| `sentrix_counterfactual_evaluations_total` | Counter | `estimator` | Total count of counterfactual simulation iterations. |

---

## Testing & Verification

The test suite enforces full test coverage, static typing, and formatting standards across over 200 unit, component, integration, and E2E lifecycle tests.

```bash
# Run unit and integration tests
pytest tests/ -v

# Run type checker
mypy --ignore-missing-imports evaluator/ api/

# Run linter and formatting checks
ruff check .
ruff format --check .
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
