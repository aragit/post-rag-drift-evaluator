"""CLI diagnostic tool for the RAG & Agent Swarm drift engine.

Subcommands:

- ``stats``: inspect telemetry store health and frame statistics.
- ``evaluate``: run ``DriftMonitor.evaluate_frames`` over two stored frame
  batches (selected by ``rag_type``) and print the drift assessment.
- ``test-alert``: dispatch a test drift payload to the webhook.

Runnable via ``python -m cli.drift_cli`` or the ``drift-cli`` entry point.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from alerting.notifier import DriftAlertNotifier
from evaluator.drift_monitor import DriftMonitor
from evaluator.drift_store import DriftStore
from evaluator.logging_config import get_logger

logger = get_logger("drift_cli")

DEFAULT_WINDOW_LIMIT = 50

TEST_ALERT_PAYLOAD: dict[str, Any] = {
    "is_drifted": True,
    "vector_drift": {"js_divergence": 0.42, "mmd_score": 0.31},
    "graph_drift": {
        "spectral_distance": 0.87,
        "density_delta": 0.12,
        "node_count_delta": 3,
    },
    "swarm_drift": {
        "transition_entropy_delta": 1.24,
        "avg_reflection_iterations_delta": 2.0,
    },
}


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


async def _cmd_stats(args: argparse.Namespace) -> int:
    store = DriftStore()
    try:
        stats = await store.get_store_stats()
    except Exception as exc:  # noqa: BLE001 - report connectivity cleanly
        logger.error("Failed to read telemetry store: %s", exc)
        return 1
    _print_json(stats)
    return 0


async def _cmd_evaluate(args: argparse.Namespace) -> int:
    store = DriftStore()
    monitor = DriftMonitor(store=store, notifier=DriftAlertNotifier())
    try:
        baseline = await store.get_recent_frames(
            rag_type=args.baseline_id, limit=args.limit
        )
        current = await store.get_recent_frames(
            rag_type=args.current_id, limit=args.limit
        )
    except Exception as exc:  # noqa: BLE001 - report connectivity cleanly
        logger.error("Failed to load stored frame batches: %s", exc)
        return 1

    result = await monitor.evaluate_frames(baseline, current)
    _print_json(result)
    return 0


async def _cmd_test_alert(args: argparse.Namespace) -> int:
    notifier = DriftAlertNotifier(webhook_url=args.webhook_url)
    if not notifier.webhook_url:
        print(
            "No webhook URL configured. Set DRIFT_ALERT_WEBHOOK_URL or pass "
            "--webhook-url.",
            file=sys.stderr,
        )
        return 1
    dispatched = await notifier.notify_if_drifted(
        TEST_ALERT_PAYLOAD, batch_id="cli-test-alert"
    )
    if not dispatched:
        logger.error("Test alert dispatch failed.")
        return 1
    _print_json(
        {
            "status": "alert dispatched",
            "webhook_url": notifier.webhook_url,
            "payload": notifier.build_payload(
                TEST_ALERT_PAYLOAD, batch_id="cli-test-alert"
            ),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drift-cli",
        description="RAG & Agent Swarm drift diagnostics and alerting tool",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats = subparsers.add_parser(
        "stats", help="Inspect telemetry store health and frame statistics"
    )
    stats.set_defaults(handler=_cmd_stats)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate drift between two stored frame batches"
    )
    evaluate.add_argument(
        "--baseline-id",
        required=True,
        help="rag_type selecting the baseline frame batch",
    )
    evaluate.add_argument(
        "--current-id",
        required=True,
        help="rag_type selecting the current frame batch",
    )
    evaluate.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_WINDOW_LIMIT,
        help=f"max frames per window (default {DEFAULT_WINDOW_LIMIT})",
    )
    evaluate.set_defaults(handler=_cmd_evaluate)

    test_alert = subparsers.add_parser(
        "test-alert", help="Dispatch a test drift alert to the configured webhook"
    )
    test_alert.add_argument(
        "--webhook-url",
        default=None,
        help="webhook URL to dispatch to (overrides DRIFT_ALERT_WEBHOOK_URL)",
    )
    test_alert.set_defaults(handler=_cmd_test_alert)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return asyncio.run(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
