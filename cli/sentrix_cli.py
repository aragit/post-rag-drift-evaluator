"""Sentrix Evaluator — CLI entrypoint.

Usage:
    sentrix --store history.jsonl --metric js_divergence
"""

from __future__ import annotations

import argparse

from evaluator.causal.attribution import attribute_drift
from evaluator.counterfactual.simulator import run_counterfactual_analysis
from evaluator.optimization.optimizer import generate_optimization_plan
from evaluator.storage import JSONHistoryStore
from evaluator.temporal.drift_detection import detect_drift_from_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentrix",
        description="Sentrix Evaluator — causal evaluation & optimization engine",
    )
    parser.add_argument(
        "--store",
        required=True,
        help="Path to the JSONL history store",
    )
    parser.add_argument(
        "--metric",
        default="js_divergence",
        help="Metric name to analyze (default: js_divergence)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Sliding window size for drift detection (default: 3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Drift detection threshold (default: 0.15)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top causal factors to simulate (default: 3)",
    )

    args = parser.parse_args(argv)

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

    print(f"      Found {len(events)} drift event(s).")

    drift_event = events[0]
    print(f"[2/4] Attributing root causes for drift event {drift_event.event_id}...")
    attribution = attribute_drift(drift_event, store)
    print(f"      Found {len(attribution.factors)} causal factor(s).")

    print("[3/4] Running counterfactual simulations...")
    counterfx = run_counterfactual_analysis(
        drift_event=drift_event,
        attribution=attribution,
        store=store,
        top_k=args.top_k,
    )
    print(f"      Generated {len(counterfx)} counterfactual result(s).")

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


if __name__ == "__main__":
    raise SystemExit(main())
