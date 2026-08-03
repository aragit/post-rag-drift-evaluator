"""Sentrix Evaluator — CLI entrypoint.

Usage:
    sentrix eval --store history.jsonl --input records.json
    sentrix drift --store history.jsonl --metric js_divergence --threshold 0.15
    sentrix remediate --store history.jsonl --metric js_divergence --threshold 0.15

Commands:
    eval       Ingest evaluation records from JSON file into the history store.
    drift      Detect drift events from the history store.
    remediate  Run full optimization pipeline: drift → attribution → CF → optimization.
"""

from __future__ import annotations

import argparse
import json

from evaluator.causal.attribution import attribute_drift
from evaluator.counterfactual.simulator import run_counterfactual_analysis
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
        print(f"  - {event.metric_name}: magnitude={event.magnitude:.4f}, "
              f"run_ids={event.involved_run_ids[:3]}")

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
    remediate_parser = subparsers.add_parser("remediate", help="Run optimization pipeline")
    remediate_parser.add_argument("--store", required=True, help="Path to JSONL store")
    remediate_parser.add_argument("--metric", default="js_divergence", help="Metric name")
    remediate_parser.add_argument("--window-size", type=int, default=3, help="Window size")
    remediate_parser.add_argument("--threshold", type=float, default=0.15, help="Threshold")
    remediate_parser.add_argument("--top-k", type=int, default=3, help="Top K factors")
    remediate_parser.set_defaults(func=cmd_remediate)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
