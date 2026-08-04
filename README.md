<div align="center">

# Sentrix Post-RAG Drift Evaluator 🛡️
Zero-Disk, Closed-Loop Latent Drift Remediation for Production RAG & LLM Swarms

[![Version](https://img.shields.io/badge/release-v0.6.1-blue.svg)](https://github.com/aragit/post-rag-drift-evaluator/releases/tag/v0.6.1)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/tests-423%20passed-success.svg)](#testing--verification)

</div>

**Sentrix Evaluator** is an enterprise-grade observability and dynamic remediation engine built to eliminate silent performance decay in Retrieval-Augmented Generation (RAG), multi-modal workflows, and LLM agent swarms.

By coupling continuous non-parametric embedding space monitoring (MMD, SWD, FD-JSD) with Bayesian causal fault attribution, Sentrix isolates vector and generation drift at the layer level. It executes zero-disk counterfactual impact simulations and policy-gated mitigation in sub-millisecond cycles—preventing system failure without adding disk I/O overhead or triggering unbounded operational feedback loops.

---

## 🚀 What's New in v0.6.1

* **Non-Blocking Asynchronous Engine:** Fully non-blocking LLM execution via `await litellm.acompletion()` and `await litellm.aembedding()` wrapped in `tenacity.AsyncRetrying`.
* **Task-Isolated Observability:** Correlation tracing using `contextvars.ContextVar` ensuring `X-Request-ID` isolation across concurrent async worker tasks.
* **Leak-Proof Connection Pooling:** Strict `async with pool.acquire()` context management across all database, Redis, and vector retrieval access paths.
* **Statistical Precision Engine:** Freedman-Diaconis dynamic binning for JSD, explicit $dx$ integration steps for continuous KDE KL divergence, and bootstrap baseline calibration.
* **Canonical `RAGRun` Telemetry Bridge:** Automated coercion of high-dimensional NumPy arrays (`query_embedding`, `retrieved_doc_embeddings`, `answer_embedding`) to JSON/JSONB-serializable evaluation frames with full pipeline provenance.

---

## Key Features

* **Dual-Track & Multi-Modal Latent Drift Detection:** Layer-separated divergence scoring across `"retrieval"` (query/document vector alignment) and `"generation"` (prompt/token output semantics) tracks, supporting both single-modality and cross-modal embedding spaces.
* **Statistically Precise Divergence Metrics:** Real-time distance evaluation using Maximum Mean Discrepancy (MMD), Sliced Wasserstein Distance (SWD), Freedman-Diaconis Jensen-Shannon Divergence (JSD), and Continuous KDE KL Divergence.
* **Canonical Data Model (`RAGRun`):** Unified domain model bridging production execution payloads to evaluation metrics, embedding state serialization, and persistent PostgreSQL `JSONB` stores.
* **Adaptive Threshold & Bootstrap Calibration:** Dynamic z-score sliding bounds ($\mu_{\text{window}} + z \cdot \sigma_{\text{window}}$) clamped to $[0.05, 0.50]$ combined with bootstrap split-half baseline calibration.
* **Causal-Latent Fusion Layer:** Direct structural mapping from latent embedding space divergence to Bayesian Causal DAG node failure priors ($P(\text{Node Failure}) = \min(1.0, \text{drift\_score} \times \text{sensitivity})$).
* **Zero-Disk Counterfactual Simulation:** In-memory state machine (`InMemoryHistoryStore`) running EWMA and OLS linear trend-adjusted counterfactual estimations without disk I/O overhead.
* **Production Gateway & Telemetry:** FastAPI gateway with health (`/health`), readiness (`/readyz`), and consolidated Prometheus metrics (`/metrics`) endpoints, plus native Kubernetes manifests and Grafana dashboard specs.

---

## System Architecture

```text
  +-----------------------------------------------------------------------------------+
  |                             RAG Pipeline Execution                                |
  |           (Queries, Context Chunks, LLM Generation, Multi-Modal Embeddings)       |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                           Canonical RAGRun Bridge                                 |
  |                (.to_evaluation_frame() / Provenance Metadata)                     |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                          StreamingDriftBuffer & Redis                             |
  |                      (Reservoir / FIFO Async Ring Buffer)                         |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                           LatentDriftEngine (Async)                               |
  |       +-----------------------------------+-----------------------------------+   |
  |       |        Retrieval Track            |        Generation Track           |   |
  |       |   (MMD / SWD / FD-JSD / KDE-KL)   |   (MMD / SWD / FD-JSD / KDE-KL)   |   |
  |       +-----------------------------------+-----------------------------------+   |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                         AdaptiveThresholdManager                                  |
  |               (Rolling Z-Score Bounds & Bootstrap Baseline)                        |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                         CausalLatentFusionEngine                                  |
  |                   (Updates Bayesian Causal DAG Node Priors)                       |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                       Zero-Disk Simulation & Guardrails                           |
  |             (InMemoryHistoryStore + EWMA/OLS & PolicyEvaluator Controls)          |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                   Prometheus Metrics & Observability Gateway                      |
  |                  (Port 8000: GET /metrics, /health, /readyz)                      |
  +-----------------------------------------------------------------------------------+
```

### Canonical Data Model: `RAGRun`

The RAGRun schema acts as the canonical data bridge between production RAG applications and the evaluation engine. It encapsulates input queries, retrieved document chunks, generated answers, associated high-dimensional vector embeddings, and operational metadata.

| Attribute | Field Type | Description |
|---|---|---|
| `run_id` | `str` | Unique UUID for the execution instance. |
| `query` | `str` | User prompt or input query string. |
| `retrieved_contexts` | `List[str]` | List of textual context chunks retrieved from vector store. |
| `generated_answer` | `str` | Final LLM response string. |
| `query_embedding` | `Optional[np.ndarray]` | High-dimensional dense vector representing the input query. |
| `retrieved_doc_embeddings` | `Optional[List[np.ndarray]]` | Array of dense vectors for retrieved context chunks. |
| `answer_embedding` | `Optional[np.ndarray]` | Dense vector representation of the generated output. |
| `metadata` | `Dict[str, Any]` | Execution provenance (pipeline_name, model, embedding_model, token_usage). |

**Coercion & Persistence Bridge:** `RAGRun.to_evaluation_frame(scores=None)` converts raw runtime data into a JSON/JSONB-serializable evaluation record, converting NumPy arrays via `.tolist()` for seamless storage in PostgreSQL `JSONB` columns:

```python
import numpy as np
from evaluator.schema import RAGRun

# Construct a canonical RAGRun instance
run = RAGRun(
    run_id="run_8f3a10b",
    query="What is the treatment for hypertension?",
    retrieved_contexts=["First-line agents include ACE inhibitors, ARBs, and CCBs."],
    generated_answer="Hypertension is typically managed using ACE inhibitors or CCBs.",
    query_embedding=np.random.randn(1536),
    retrieved_doc_embeddings=[np.random.randn(1536)],
    answer_embedding=np.random.randn(1536),
    metadata={
        "pipeline_name": "medical_rag_v2",
        "model": "google/MedGemma-4B-IT",
        "embedding_model": "text-embedding-3-large",
        "token_usage": {"prompt_tokens": 120, "completion_tokens": 45}
    }
)

# Convert to JSONB-safe evaluation frame
eval_frame = run.to_evaluation_frame(scores={"relevance": 0.92, "faithfulness": 0.88})
```

---

## Drift Detection Methodology & Algorithms

Sentrix implements four primary statistical divergence engines designed for high-dimensional latent space monitoring:

1. **Maximum Mean Discrepancy (MMD):** Computes non-parametric distance between distributions $P$ and $Q$ using a Radial Basis Function (RBF) kernel with median heuristic bandwidth scaling ($\sigma$):

$$k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

$$\text{MMD}^2(P, Q) = \mathbb{E}[k(x, x')] - 2\mathbb{E}[k(x, y)] + \mathbb{E}[k(y, y')]$$

2. **Sliced Wasserstein Distance (SWD):** Projects high-dimensional vectors onto $L$ random 1D unit vectors $\{\theta_l\}_{l=1}^L$ and averages the 1D Wasserstein distances across projections:

$$\text{SWD}(P, Q) = \frac{1}{L} \sum_{l=1}^L \mathcal{W}_1(\theta_l \sharp P, \theta_l \sharp Q)$$

3. **Freedman-Diaconis Jensen-Shannon Divergence (JSD):** To eliminate sensitivity to fixed bin sizes, JSD uses dynamic Freedman-Diaconis bin width estimation over shared Principal Component Analysis (PCA) projection axes:

$$h = 2 \cdot \frac{\text{IQR}(X)}{n^{1/3}} \implies K = \text{clamp}\left(\left\lceil \frac{\max(X) - \min(X)}{h} \right\rceil, 10, 100\right)$$

$$\text{JSD}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{1}{2}(P + Q)$$

4. **Continuous KDE Kullback-Leibler (KL) Divergence:** Computes continuous KL divergence on Kernel Density Estimations with explicit integration step size ($dx = x_{i+1} - x_i$) and zero-division protection ($\epsilon = 1e-12$):

$$D_{KL}(P \| Q) = \sum_{i=1}^N P(x_i) \log\left( \frac{P(x_i) + \epsilon}{Q(x_i) + \epsilon} \right) \cdot dx$$

---

## Python SDK Reference

### 1. Streaming Buffer & Dynamic Adaptive Drift Detection

```python
import asyncio
import numpy as np
from evaluator.latent_drift import (
    StreamingDriftBuffer,
    AdaptiveThresholdManager,
    LatentDriftEngine,
)

async def main():
    # Initialize streaming buffer and dynamic threshold manager
    buffer = StreamingDriftBuffer(capacity=1000, sample_strategy="reservoir")
    threshold_mgr = AdaptiveThresholdManager(base_threshold=0.15, sensitivity_z=2.0)
    engine = LatentDriftEngine(threshold_manager=threshold_mgr)

    # Ingest high-dimensional query embeddings from real-time stream
    vectors = np.random.randn(100, 1536)
    for vec in vectors:
        await buffer.aingest(vector=vec, track="retrieval")

    # Flush current buffer snapshot and evaluate dynamic latent drift
    if buffer.is_ready(min_samples=50):
        batch = await buffer.aflush_batch(track="retrieval")
        drift_result = await engine.acompute_drift(
            current_batch=batch,
            baseline_batch=batch,
            method="jsd"
        )
        print(f"Drift Score: {drift_result.score:.4f}")
        print(f"Dynamic Threshold Used: {drift_result.metadata['dynamic_threshold']:.4f}")

asyncio.run(main())
```

### 2. Zero-Disk Closed-Loop Optimization

```python
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
| `/health` | GET | Service liveness probe. | _None_ |
| `/readyz` | GET | Service readiness check (DB / Redis connection verification). | _None_ |
| `/metrics` | GET | Consolidated Prometheus metrics scrape endpoint. | _None_ |
| `/v1/eval` | POST | Ingest raw EvaluationRecord / RAGRun batches into storage. | `{"records": [...]}` |
| `/v1/drift/detect` | POST | Triggers dual-track MMD/SWD/JSD detection across batches. | `{"current_batch_id": "...", "track": "retrieval"}` |
| `/v1/remediate` | POST | Executes full OptimizationRunner remediation cycle. | `{"drift_event": {...}}` |
| `/v1/stream/ingest` | POST | Ingests a single vector into the live streaming buffer. | `{"vector": [...], "track": "retrieval"}` |
| `/v1/stream/flush` | POST | Flushes stream buffer and evaluates current drift. | `{"track": "retrieval", "metric": "jsd"}` |

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

## 📊 Telemetry & Observability Stack (OpenTelemetry + Prometheus + Grafana)

Sentrix native telemetry exposes operational metrics via standard OpenTelemetry meters with automatic fallbacks to Prometheus HTTP exporter primitives. The repository includes a zero-click Docker setup with pre-configured scrape targets and auto-provisioned Grafana dashboards (`dashboards/grafana-sentrix-overview.json`).

---

### 🚀 Zero-Click Quick Start

Launch the monitoring control plane with pre-loaded dashboards and data sources:

```bash
# 1. Start Prometheus & Grafana with auto-provisioning
docker compose up -d

# 2. Stream live synthetic telemetry metrics
python scripts/stream_metrics.py
```

Access Grafana at http://localhost:3000 (Credentials: `admin` / `visdrift` — Dashboard is auto-provisioned on startup)

---

### ☸️ Kubernetes Auto-Discovery & Pod Annotations

For Kubernetes deployment topologies, add these annotations to pod metadata to enable Prometheus scrape auto-discovery:

```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/path: "/metrics"
    prometheus.io/port: "8000"
```

---

### ⚙️ Manual Container Execution (Alternative)

If executing container instances individually without Docker Compose:

```bash
# Launch Prometheus
docker run -d --name prometheus -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Launch Grafana (Linux requires host gateway flag)
docker run -d --name grafana -p 3000:3000 \
  --add-host=host.docker.internal:host-gateway \
  -e "GF_SECURITY_ADMIN_PASSWORD=visdrift" \
  -v $(pwd)/grafana/provisioning:/etc/grafana/provisioning \
  -v $(pwd)/dashboards:/var/lib/grafana/dashboards \
  grafana/grafana
```

Cross-Platform Note: Docker Desktop (macOS/Windows) automatically resolves `host.docker.internal`. Native Linux environments utilize the `--add-host=host.docker.internal:host-gateway` flag defined in `docker-compose.yml`.

---

### 📈 Exposed Telemetry Metrics Reference

| Metric Name | Type | Labels / Dimensions | Description |
|---|---|---|---|
| `sentrix_up` | Gauge | _pid_ | Evaluator API process status (1 = running, 0 = down). |
| `sentrix_drift_score` | Gauge | `metric_type` | Real-time calculated drift metric score (vector_jsd, graph, swarm). |
| `sentrix_evaluation_duration_seconds` | Histogram | `status` | Multi-modal drift evaluation latency profile (P50 / P95 / P99). |
| `frame_ingestion_total` | Counter | `status`, `buffer_type` | Total telemetry frames ingested via the HTTP API. |
| `ingestion_buffer_depth` | Gauge | `buffer_type` | Ingestion queue depth awaiting persistence. |
| `drift_alerts_total` | Counter | `status` | Total drift alerts dispatched by the governance gate. |
| `db_batch_write_latency_seconds` | Histogram | _(none)_ | Latency profile for database batch write operations. |

---

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/aragit/post-rag-drift-evaluator.git
cd post-rag-drift-evaluator

# Install core package
pip install -e .

# Install with development and UI extras
pip install -e ".[dev,ui]"
```

### Docker Deployment

```bash
# Build multi-stage image
docker build -t sentrix-evaluator:v0.6.1 .

# Spin up production stack
docker compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
# Apply secrets, configmap, deployment, and service
kubectl apply -f k8s/deployment.yaml

# Apply Grafana dashboard sidecar ConfigMap
kubectl apply -f k8s/grafana-dashboard-configmap.yaml
```

### Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string (`sslmode=require` in prod). |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis instance URL for stream ingestion buffering. |
| `LITELLM_MASTER_KEY` | `""` | Master API key for LLM proxy calls. |
| `LOG_LEVEL` | `INFO` | System logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `STREAM_CAPACITY` | `1000` | Sliding window capacity for `StreamingDriftBuffer`. |
| `DRIFT_THRESHOLD_Z` | `2.0` | Z-score multiplier for dynamic threshold calculation. |

---

## Troubleshooting Matrix

| Problem | Cause | Resolution |
|---|---|---|
| Timeout reading from `redis:6379` | Idle Redis ingestion stream or network pause. | Expected behavior when idle. System automatically falls back to `AsyncIngestionBuffer`. |
| `kind load docker-image` hangs or fails | Docker snap `/tmp` mount namespace conflict or low disk space. | Use direct streaming pipe: `docker save img \| docker exec -i node ctr -n k8s.io images import -`. |
| Port 8000 collision on startup | Standalone Prometheus HTTP server running alongside FastAPI. | v0.6.1 consolidates `/metrics` directly onto FastAPI app on port 8000. Stop legacy exporters. |
| Database Connection Pool Exhausted | Unclosed connections in async tasks. | Wrap all DB operations in `async with pool.acquire() as conn:`. |

---

## Testing & Verification

The test suite enforces full test coverage, static typing, and formatting standards across over 400 unit, component, integration, and E2E lifecycle tests.

```bash
# Execute pre-flight check script
./scripts/pre_flight_check.sh

# Run unit and integration tests
pytest tests/ -v

# Run type checker
mypy --ignore-missing-imports evaluator/ api/

# Run linter and formatting checks
ruff check .
ruff format --check .

# Run live API smoke test
python tests/smoke_test.py --url http://localhost:8000
```

---

## Citation

If you use Sentrix Evaluator in your research or production systems, please cite it as follows:

```bibtex
@software{sentrix_evaluator_2026,
  author = {Nicoomanesh, Arash},
  title = {Sentrix Evaluator: Enterprise Zero-Disk Closed-Loop Drift Remediation Engine for RAG and Multi-Modal Pipelines},
  year = {2026},
  version = {0.6.1},
  url = {https://github.com/aragit/post-rag-drift-evaluator}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
