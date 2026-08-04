#!/usr/bin/env python3
"""Continuously generate realistic varying telemetry metrics for Grafana testing.

Sends periodic POST requests to the Sentrix API's /v1/telemetry/evaluate
endpoint with varying embeddings, graph topologies, and agent hops to
produce dynamic drift metrics visible in Grafana dashboards.

Usage:
    python scripts/stream_metrics.py [--interval 2] [--duration 60]

Requirements: httpx (pip install httpx)
"""

import argparse
import random
import time
from uuid import uuid4

import httpx

BASE_URL = "http://localhost:8000"

PATH_GRAPH = {
    "nodes": [{"id": str(i)} for i in range(1, 5)],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "4"},
    ],
}

CYCLE_GRAPH = {
    "nodes": [{"id": str(i)} for i in range(1, 5)],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "4"},
        {"source": "4", "target": "1"},
    ],
}


def make_frame(idx: int, *, drift_factor: float = 0.0) -> dict:
    """Generate a realistic RAG evaluation frame with optional drift.

    ``drift_factor`` controls how much the embedding deviates from the
    baseline, simulating realistic vector drift scenarios.
    """
    emb = [
        0.1 + random.gauss(0, 0.01) + drift_factor * random.uniform(-1, 1),
        0.2 + random.gauss(0, 0.01) + drift_factor * random.uniform(-1, 1),
        0.01 * idx,
    ]
    rag_type = random.choice(["naive", "agentic", "graph_rag", "swarm"])

    graph = None
    agent_hops = None
    reflection_iterations = 0

    if rag_type == "graph_rag":
        graph = random.choice([PATH_GRAPH, CYCLE_GRAPH])
    elif rag_type == "swarm":
        agent_hops = random.choices(
            ["alpha", "beta", "gamma", "delta"], k=random.randint(2, 6)
        )
        reflection_iterations = random.randint(0, 5)
    elif rag_type == "agentic":
        agent_hops = random.choices(
            ["a", "b", "c", "d"], k=random.randint(2, 5)
        )
        reflection_iterations = random.randint(0, 3)

    return {
        "trace_id": str(uuid4()),
        "query": {
            "text": f"Query #{idx}: What is the causal impact of embedding drift on RAG performance?",
            "embedding": emb,
        },
        "context": {
            "text_chunks": [
                "Drift detection monitors embedding distribution shifts.",
                "Counterfactuals simulate policy impact.",
                "Causal attribution maps latent divergence to DAG node priors.",
            ],
            "graph_topology": graph,
            "metadata": {"retriever": "faiss", "top_k": 5},
        },
        "metadata": {
            "rag_type": rag_type,
            "agent_hops": agent_hops,
            "reflection_iterations": reflection_iterations,
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "pipeline_name": random.choice(
                ["medical_rag_v2", "finance_qa", "legal_assistant", "swarm_coordinator"]
            ),
        },
        "output": {
            "generated_answer": f"Drift score {random.uniform(0, 1):.4f} indicates "
            f"{'high' if random.random() > 0.5 else 'low'} divergence.",
            "confidence_score": random.uniform(0.7, 0.99),
            "response_embedding": emb,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously stream realistic evaluation metrics to Grafana."
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Seconds between requests (default: 2)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Total duration in seconds (0 = unlimited, default: 0)",
    )
    args = parser.parse_args()

    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        try:
            health = client.get("/health")
            print(f"Health check: {health.status_code}")
        except Exception as e:
            print(f"Health check failed: {e}")
            return

        start_time = time.monotonic()
        request_count = 0

        print(f"Streaming metrics every {args.interval}s"
              f"{f' for {args.duration}s' if args.duration > 0 else ' (unlimited)'}...")
        try:
            while True:
                if args.duration > 0 and (time.monotonic() - start_time) >= args.duration:
                    print(f"\nDuration reached. Total requests sent: {request_count}")
                    break

                # Gradually increase drift to create visible metric trends
                elapsed = time.monotonic() - start_time
                drift_factor = min(0.5, elapsed / 30.0)

                baseline = [make_frame(j, drift_factor=0.01) for j in range(3)]
                current = [make_frame(request_count + j, drift_factor=drift_factor) for j in range(3)]

                r = client.post(
                    "/v1/telemetry/evaluate",
                    json={"baseline_frames": baseline, "current_frames": current},
                )
                is_drifted = r.json().get("is_drifted", False)
                print(f"  [{request_count + 1}] {r.status_code} drifted={is_drifted}")
                request_count += 1

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\nStopped. Total requests sent: {request_count}")


if __name__ == "__main__":
    main()
