"""Sentrix Evaluator — CLI entrypoint.

Usage:
    sentrix eval --store history.jsonl --input records.json
    sentrix drift --store history.jsonl --metric js_divergence --threshold 0.15
    sentrix remediate --store history.jsonl --metric js_divergence --threshold 0.15
    sentrix stream --track retrieval --capacity 500 < vectors.jsonl

Commands:
    eval       Ingest evaluation records from JSON file into the history store.
    drift      Detect drift events from the history store.
    remediate  Run full optimization pipeline: drift → attribution → CF → optimization.
    stream     Ingest embedding vectors from stdin/file into a streaming buffer
               and optionally flush for drift detection.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from evaluator.causal.attribution import attribute_drift
from evaluator.counterfactual.simulator import run_counterfactual_analysis
from evaluator.latent_drift import (
    EmbeddingBatch,
    StreamingDriftBuffer,
    compute_latent_drift,
)
from evaluator.optimization.optimizer import generate_optimization_plan
from evaluator.storage import JSONHistoryStore
from evaluator.temporal.drift_detection import detect_drift_from_store


def cmd_eval(args: argparse.Namespace) -> int:
    """Ingest evaluation records from a JSON file into the history store."""
    store = JSONHistoryStore(args.store)

    with open(args.input) as f:
        records = json.load(f)

    if isinstance(records, dict):
        records = [records]

    count = 0
    from evaluator.storage.models import EvaluationRecord

    for record_data in records:
        record = EvaluationRecord.from_dict(record_data)
        store.save(record)
        count += 1

    print(f"Ingested {count} record(s) into {args.store}")
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    """Detect drift events from the history store."""
    store = JSONHistoryStore(args.store)

    events = detect_drift_from_store(
        store=store,
        metric_name=args.metric,
        window_size=args.window_size,
        threshold=args.threshold,
    )

    if not events:
        print("No drift detected. No optimization needed.")
        return 0

    print(f"Found {len(events)} drift event(s):")
    for event in events:
        print(
            f"  - {event.metric_name}: magnitude={event.magnitude:.4f}, "
            f"run_ids={event.involved_run_ids[:3]}"
        )

    return 0


def cmd_remediate(args: argparse.Namespace) -> int:
    """Run full optimization pipeline: drift → attribution → CF → optimization."""
    store = JSONHistoryStore(args.store)

    print(f"[1/4] Detecting drift on metric '{args.metric}'...")
    events = detect_drift_from_store(
        store=store,
        metric_name=args.metric,
        window_size=args.window_size,
        threshold=args.threshold,
    )

    if not events:
        print("No drift detected. No optimization needed.")
        return 0

    drift_event = events[0]
    print(f"      Found drift event {drift_event.event_id}")

    print("[2/4] Attributing root causes...")
    attribution = attribute_drift(drift_event, store)
    print(f"      Found {len(attribution.factors)} causal factor(s)")

    print("[3/4] Running counterfactual simulations...")
    counterfx = run_counterfactual_analysis(
        drift_event=drift_event,
        attribution=attribution,
        store=store,
        top_k=args.top_k,
    )
    print(f"      Generated {len(counterfx)} counterfactual result(s)")

    print("[4/4] Generating optimization plan...")
    plan = generate_optimization_plan(drift_event, attribution, counterfx)
    print()
    print(plan.summary)
    print()

    if plan.recommendations:
        print("Recommendations:")
        for rec in plan.recommendations:
            print(
                f"  #{rec.priority} [{rec.action.action_type}] "
                f"→ {rec.action.description} "
                f"(improvement: {rec.expected_improvement:.4f}, "
                f"confidence: {rec.confidence:.2f})"
            )
    else:
        print("No recommendations generated.")

    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    """Ingest embedding vectors from stdin/file into a streaming buffer and optionally flush."""
    buffer = StreamingDriftBuffer(
        capacity=args.capacity,
        sample_strategy=args.strategy,
    )

    lines_read = 0
    if args.input and args.input != "-":
        with open(args.input) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        vector = data.get("vector", data)
        track = data.get("track", args.track)
        buffer.ingest(np.array(vector, dtype=float), track=track)
        lines_read += 1

    print(
        f"Ingested {lines_read} vector(s) into streaming buffer (capacity={args.capacity})"
    )
    print(f"Buffer sizes: {buffer.sizes}")

    if args.flush:
        current_batch = buffer.flush_batch()
        print(f"Flushed {current_batch.vectors.shape[0]} vector(s)")

        if current_batch.vectors.shape[0] == 0:
            print("Buffer empty, nothing to flush.")
            return 0

        track = args.track
        track_batch = buffer.flush_track(track=track)
        if track_batch.vectors.shape[0] == 0:
            print(f"No data for track '{track}'")
            return 0

        baseline = EmbeddingBatch(
            vectors=np.random.RandomState(42).normal(
                0, 1, size=(100, track_batch.vectors.shape[1])
            ),
            track=track,
        )
        current = EmbeddingBatch(
            vectors=track_batch.vectors,
            track=track,
        )

        result = compute_latent_drift(
            baseline=baseline,
            current=current,
            threshold=args.threshold,
            metric=args.metric,
            pca_components=min(5, track_batch.vectors.shape[1]),
        )
        print(
            f"Drift score: {result.drift_score:.4f} (threshold={result.threshold:.4f})"
        )
        print(f"Drift detected: {result.drift_detected}")
        print(f"Metric: {result.metric_used}, Track: {result.track}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentrix",
        description="Sentrix Evaluator — causal evaluation & optimization engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Ingest evaluation records")
    eval_parser.add_argument(
        "--store", required=True, help="Path to the JSONL history store"
    )
    eval_parser.add_argument(
        "--input", required=True, help="Path to JSON file with record(s) to ingest"
    )
    eval_parser.set_defaults(func=cmd_eval)

    # drift command
    drift_parser = subparsers.add_parser("drift", help="Detect drift events")
    drift_parser.add_argument("--store", required=True, help="Path to JSONL store")
    drift_parser.add_argument("--metric", default="js_divergence", help="Metric name")
    drift_parser.add_argument("--window-size", type=int, default=3, help="Window size")
    drift_parser.add_argument("--threshold", type=float, default=0.15, help="Threshold")
    drift_parser.set_defaults(func=cmd_drift)

    # remediate command
    remediate_parser = subparsers.add_parser(
        "remediate", help="Run optimization pipeline"
    )
    remediate_parser.add_argument("--store", required=True, help="Path to JSONL store")
    remediate_parser.add_argument(
        "--metric", default="js_divergence", help="Metric name"
    )
    remediate_parser.add_argument(
        "--window-size", type=int, default=3, help="Window size"
    )
    remediate_parser.add_argument(
        "--threshold", type=float, default=0.15, help="Threshold"
    )
    remediate_parser.add_argument("--top-k", type=int, default=3, help="Top K factors")
    remediate_parser.set_defaults(func=cmd_remediate)

    # stream command
    stream_parser = subparsers.add_parser(
        "stream", help="Ingest streaming embedding vectors"
    )
    stream_parser.add_argument(
        "--capacity", type=int, default=1000, help="Buffer capacity"
    )
    stream_parser.add_argument(
        "--strategy",
        choices=["reservoir", "fifo"],
        default="reservoir",
        help="Buffer overflow strategy",
    )
    stream_parser.add_argument(
        "--track",
        default="retrieval",
        choices=["retrieval", "generation"],
        help="Embedding track",
    )
    stream_parser.add_argument(
        "--metric", default="mmd", help="Drift metric (mmd, swd, jsd)"
    )
    stream_parser.add_argument(
        "--threshold", type=float, default=0.15, help="Drift threshold"
    )
    stream_parser.add_argument(
        "--flush",
        action="store_true",
        help="Flush after ingestion and run drift detection",
    )
    stream_parser.add_argument(
        "--input",
        default="-",
        help="Path to JSONL file (default: stdin, use '-' for stdin)",
    )
    stream_parser.set_defaults(func=cmd_stream)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
