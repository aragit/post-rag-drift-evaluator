#!/usr/bin/env python3
"""End-to-end integration verification for the RAG drift engine.

Attempts to spin up the full container stack via ``docker compose``.
If the Docker image build is unavailable (e.g. no network), falls back
to running the API locally via ``uvicorn`` while still using PostgreSQL
and Redis containers for backing services.

After startup the script:
  1. Polls ``/healthz`` and ``/readyz`` (exponential backoff, 45 s timeout).
  2. Ingests a batch of ``RAGEvaluationFrame`` payloads.
  3. Waits for the Redis Stream background worker to flush to PostgreSQL.
  4. Calls ``POST /evaluate`` with **no explicit baseline** — exercises the
     dynamic sliding-baseline auto-calibration path.
  5. Queries ``GET /metrics`` and asserts required Prometheus series exist.
  6. Tears down all containers and the API subprocess.
"""

from __future__ import annotations

import os
import random
import signal
import subprocess
import sys
import time
from typing import Any

import httpx
import numpy as np

from ingestion.run_schema import RAGRun

# ── Docker environment ──────────────────────────────────────────────────
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_DB", "rag_db")
os.environ.setdefault("POSTGRES_PORT", "5433")

# API connection
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
HEALTH_TIMEOUT_S = 45
POLL_BACKOFF_S = 0.5
POLL_MAX_INTERVAL_S = 3
FRAME_FLUSH_DELAY_S = int(os.environ.get("FRAME_FLUSH_DELAY_S", "3"))

# ── Test data ───────────────────────────────────────────────────────────

GRAPH_PATH: dict[str, Any] = {
    "nodes": [{"id": str(i)} for i in range(1, 5)],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "4"},
    ],
}

GRAPH_CYCLE: dict[str, Any] = {
    "nodes": [{"id": str(i)} for i in range(1, 5)],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "4"},
        {"source": "4", "target": "1"},
    ],
}


def make_frame(
    idx: int, embedding_dim: int = 8, seed: int | None = None
) -> dict[str, Any]:
    """Build a realistic ``RAGEvaluationFrame`` JSON payload."""
    rng = random.Random(seed if seed is not None else idx)
    embedding = [rng.gauss(0, 1) for _ in range(embedding_dim)]
    return {
        "query": {"text": f"question_{idx}", "embedding": embedding},
        "context": {
            "text_chunks": [f"chunk_{idx}_a", f"chunk_{idx}_b"],
            "dense_embeddings": [embedding],
            "graph_topology": GRAPH_PATH if idx % 2 == 0 else GRAPH_CYCLE,
        },
        "metadata": {
            "rag_type": "agentic",
            "agent_hops": ["retriever", "reader", "retriever"],
            "reflection_iterations": 1,
            "latency_ms": 120.0 + idx,
        },
        "output": {
            "generated_answer": f"answer_{idx}",
            "confidence_score": 0.85 + idx * 0.01,
        },
    }


def make_ragrun(idx: int, embedding_dim: int = 8, seed: int | None = None) -> RAGRun:
    """Build a canonical :class:`RAGRun` from the same test data as :func:`make_frame`."""
    from ingestion.run_schema import RAGSystemInfo

    rng = random.Random(seed if seed is not None else idx)
    embedding = [rng.gauss(0, 1) for _ in range(embedding_dim)]
    embedding_arr = np.asarray(embedding, dtype=float)
    doc_a_emb = np.asarray([rng.gauss(0, 1) for _ in range(embedding_dim)], dtype=float)
    doc_b_emb = np.asarray([rng.gauss(0, 1) for _ in range(embedding_dim)], dtype=float)
    return RAGRun(
        query=f"question_{idx}",
        retrieved_docs=[f"chunk_{idx}_a", f"chunk_{idx}_b"],
        retrieved_doc_ids=[f"doc_{idx}_a", f"doc_{idx}_b"],
        retrieved_embeddings=[doc_a_emb, doc_b_emb],
        query_embedding=embedding_arr,
        answer=f"answer_{idx}",
        answer_embedding=embedding_arr,
        system_info=RAGSystemInfo(
            name="AgenticRAG",
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            retriever="vector",
            version="0.1.0",
        ),
        metadata={
            "rag_type": "agentic",
            "agent_hops": ["retriever", "reader", "retriever"],
            "reflection_iterations": 1,
            "latency_ms": 120.0 + idx,
            "confidence_score": 0.85 + idx * 0.01,
        },
    )


def make_frame_batch(n: int) -> list[dict[str, Any]]:
    return [make_frame(i, seed=42 + i) for i in range(n)]


def make_ragrun_batch(n: int) -> list[RAGRun]:
    return [make_ragrun(i, seed=42 + i) for i in range(n)]


# ── Docker management ───────────────────────────────────────────────────


def _compose(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose"] + args,
        capture_output=True,
        text=True,
        cwd=_PROJECT_DIR,
    )


def start_docker_services() -> bool:
    """Start db and redis containers via docker compose.

    Returns True if containers were started, False if they already exist
    or docker compose is unavailable.
    """
    print("[1/5] Starting Docker Compose (db + redis)...")
    t0 = time.monotonic()

    # Try to start just the db and redis services (skip api — build may fail)
    result = _compose(["up", "-d", "db", "redis"])

    if result.returncode == 0:
        elapsed = time.monotonic() - t0
        print(f"  Containers started in {elapsed:.1f}s")
        # Wait a moment for services to be ready
        time.sleep(5)
        return True
    else:
        # Containers might already be running
        print(f"  docker compose up: rc={result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:200]}")
        return False


def start_api_local() -> subprocess.Popen | None:
    """Start the FastAPI API as a local uvicorn subprocess."""
    print("  Starting API locally via uvicorn...")
    env = os.environ.copy()
    db_host = os.environ.get("POSTGRES_HOST", "localhost")
    db_port = os.environ.get("POSTGRES_PORT", "5432")
    db_user = os.environ.get("POSTGRES_USER", "postgres")
    db_pass = os.environ.get("POSTGRES_PASSWORD", "postgres")
    db_name = os.environ.get("POSTGRES_DB", "rag_db")
    env["DATABASE_URL"] = (
        f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    )
    env["REDIS_URL"] = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    env["POSTGRES_HOST"] = db_host
    env["POSTGRES_PORT"] = db_port
    env["API_KEY_REQUIRED"] = "False"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=_PROJECT_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  API subprocess PID={proc.pid}")
    return proc


def compose_down() -> None:
    """Tear down Docker Compose containers and volumes."""
    print("\n[5/5] Tearing down Docker Compose stack...")
    result = _compose(["down", "-v"])
    if result.returncode == 0:
        print("  Stack torn down (--volumes removed).")
    else:
        print(f"  Warning: docker compose down rc={result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:200]}")


def stop_api(proc: subprocess.Popen | None) -> None:
    """Terminate the local API subprocess."""
    if proc is None:
        return
    print("  Stopping API subprocess...")
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    print("  API subprocess stopped.")


# ── Health & readiness polling ──────────────────────────────────────────


def poll_health() -> bool:
    """Poll /healthz then /readyz until both return 200 (exponential backoff)."""
    print("[2/5] Waiting for API health & readiness...")
    deadline = time.time() + HEALTH_TIMEOUT_S
    interval = POLL_BACKOFF_S
    health_ok = False
    attempts = 0

    while time.time() < deadline:
        attempts += 1

        # Liveness
        if not health_ok:
            try:
                r = httpx.get(f"{BASE_URL}/healthz", timeout=2)
                if r.status_code == 200:
                    health_ok = True
                    print(f"  /healthz → 200 OK  ({r.json()})")
                else:
                    print(f"  /healthz → {r.status_code}")
            except Exception as e:
                if attempts <= 3:
                    print(f"  /healthz → not reachable ({e.__class__.__name__})")

        # Readiness (only after liveness passes)
        if health_ok:
            try:
                r = httpx.get(f"{BASE_URL}/readyz", timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    print(f"  /readyz  → 200 OK  ({data})")
                    return True
                else:
                    detail = ""
                    try:
                        detail = r.json().get("detail", "")
                    except Exception:
                        pass
                    if attempts <= 3:
                        print(f"  /readyz  → {r.status_code} ({detail})")
            except Exception as e:
                if attempts <= 3:
                    print(f"  /readyz  → not reachable ({e.__class__.__name__})")

        time.sleep(interval)
        interval = min(interval * 1.5, POLL_MAX_INTERVAL_S)

    print(f"  Timed out after {HEALTH_TIMEOUT_S}s waiting for readiness.")
    return False


# ── Telemetry flow ──────────────────────────────────────────────────────


def ingest_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """POST a batch of frames and assert 202."""
    print("[3/5] Ingesting telemetry frames...")
    resp = httpx.post(
        f"{BASE_URL}/v1/telemetry/frames", json={"frames": frames}, timeout=10
    )
    assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
    body = resp.json()
    print(
        f"  POST /v1/telemetry/frames → {resp.status_code}  accepted={body.get('count')}"
    )
    return body


def wait_for_flush() -> None:
    """Allow the Redis Stream background worker to flush frames to PostgreSQL."""
    print(f"  Waiting {FRAME_FLUSH_DELAY_S}s for background worker flush...")
    time.sleep(FRAME_FLUSH_DELAY_S)


def evaluate_drift() -> dict[str, Any]:
    """POST an evaluate request without explicit baseline (dynamic fallback)."""
    print("[4/5] Evaluating drift with dynamic sliding baseline...")
    current = make_frame_batch(4)
    resp = httpx.post(
        f"{BASE_URL}/v1/telemetry/evaluate",
        json={"current_frames": current},
        timeout=15,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    result = resp.json()
    print(f"  POST /v1/telemetry/evaluate → {resp.status_code}")
    print(f"    is_drifted        = {result.get('is_drifted')}")
    print(f"    vector_jsd        = {result['vector_drift'].get('js_divergence'):.6f}")
    print(f"    vector_mmd        = {result['vector_drift'].get('mmd_score'):.6f}")
    print(
        f"    graph_spectral    = {result['graph_drift'].get('spectral_distance'):.6f}"
    )
    print(
        f"    swarm_entropy     = {result['swarm_drift'].get('transition_entropy_delta'):.6f}"
    )
    return result


def check_metrics() -> None:
    """GET /metrics and assert required Prometheus series are present."""
    print("  Checking Prometheus metrics...")
    resp = httpx.get(f"{BASE_URL}/metrics", timeout=10)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    text = resp.text

    required = [
        ("frame_ingestion_total", "FRAME_INGESTION_TOTAL"),
        ("sentrix_drift_score", "DRIFT_SCORE_GAUGE"),
        ("db_batch_write_latency_seconds", "DB_BATCH_WRITE_LATENCY_SECONDS"),
    ]
    for metric_name, label in required:
        assert metric_name in text, (
            f"Metric '{metric_name}' ({label}) not found in /metrics output"
        )
        print(f"    ✓ {label} ({metric_name}) present")

    # Spot-check labeled series
    assert 'status="accepted"' in text, (
        "frame_ingestion_total{status='accepted'} missing"
    )
    assert 'metric_type="vector_jsd"' in text, (
        "drift_score_gauge{metric_type='vector_jsd'} missing"
    )


print("    ✓ labeled series verified")


# ── Orchestration ───────────────────────────────────────────────────────


def _is_api_reachable() -> bool:
    """Return True if the API is already responding on BASE_URL."""
    try:
        r = httpx.get(f"{BASE_URL}/healthz", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    results: dict[str, Any] = {"checks": [], "summary": {}}
    api_proc: subprocess.Popen | None = None
    api_already_running = False

    try:
        # Step 1: start backing services (skipped if API already running in Docker)
        api_already_running = _is_api_reachable()
        if not api_already_running:
            start_docker_services()
            api_proc = start_api_local()
        else:
            print("  API already reachable — skipping local startup (Docker mode).")

        # Step 2: wait for readiness
        if not poll_health():
            results["summary"]["health_status"] = "FAILED"
            print("\n✗ Health/readiness check failed — aborting.")
            sys.exit(1)
        results["summary"]["health_status"] = "OK"

        # Step 3: ingest frames
        baseline_frames = make_frame_batch(8)
        ingest_response = ingest_frames(baseline_frames)
        results["summary"]["frames_ingested"] = ingest_response.get("count")
        results["checks"].append("telemetry_ingested")

        # Allow background worker to flush to PostgreSQL
        wait_for_flush()

        # Step 4: evaluate with dynamic baseline
        eval_result = evaluate_drift()
        results["summary"]["eval_result"] = {
            "is_drifted": eval_result["is_drifted"],
            "vector_jsd": eval_result["vector_drift"]["js_divergence"],
            "vector_mmd": eval_result["vector_drift"]["mmd_score"],
            "graph_spectral": eval_result["graph_drift"]["spectral_distance"],
            "swarm_entropy": eval_result["swarm_drift"]["transition_entropy_delta"],
        }
        results["checks"].append("dynamic_baseline_evaluated")

        # Step 5: check metrics
        check_metrics()
        results["checks"].append("metrics_verified")

        # Summary
        print("\n" + "=" * 60)
        print("✓ ALL VERIFICATION CHECKS PASSED")
        print("=" * 60)
        print(f"  Health status     : {results['summary']['health_status']}")
        print(f"  Frames ingested   : {results['summary']['frames_ingested']}")
        print(
            f"  is_drifted        : {results['summary']['eval_result']['is_drifted']}"
        )
        print(f"  Checks passed     : {', '.join(results['checks'])}")

    except AssertionError as e:
        print(f"\n✗ VERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e.__class__.__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        stop_api(api_proc)
        if not api_already_running:
            compose_down()


if __name__ == "__main__":
    main()
