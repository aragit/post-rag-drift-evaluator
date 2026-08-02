"""Swarm-level drift detection for multi-agent RAG outputs.

Compares the routing and reflection behavior of an agent swarm between a
baseline and a current window:

- **Transition entropy delta**: Shannon entropy of the agent-hop
  transition matrix (``-sum p_ij * ln p_ij``), capturing how
  deterministic or scattered agent routing has become.
- **Reflection iteration delta**: shift in the average number of
  reflection iterations per trace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

import numpy as np

from evaluator.logging_config import get_logger
from evaluator.schemas.telemetry import ExecutionMetadataPayload

logger = get_logger("SwarmDriftCalculator")

SwarmMeta = Union[ExecutionMetadataPayload, Dict[str, Any]]


def _agent_hops(payload: SwarmMeta) -> List[str]:
    if isinstance(payload, dict):
        return payload.get("agent_hops") or []
    return payload.agent_hops or []


def _reflection_iterations(payload: SwarmMeta) -> int:
    if isinstance(payload, dict):
        return int(payload.get("reflection_iterations", 0) or 0)
    return int(payload.reflection_iterations or 0)


def _transition_matrix(payloads: Sequence[SwarmMeta], agents: List[str]) -> np.ndarray:
    index = {agent: i for i, agent in enumerate(agents)}
    matrix = np.zeros((len(agents), len(agents)), dtype=float)
    for payload in payloads:
        hops = _agent_hops(payload)
        for source, target in zip(hops, hops[1:]):
            matrix[index[source], index[target]] += 1.0
    return matrix


def _matrix_entropy(matrix: np.ndarray) -> float:
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-12, None)
    probabilities = normalized[normalized > 0]
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)))


class SwarmDriftCalculator:
    """Compare agent-swarm routing and reflection behavior across windows."""

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        reflection_threshold: float = 1.0,
    ):
        self.entropy_threshold = entropy_threshold
        self.reflection_threshold = reflection_threshold

    def compute_swarm_drift(
        self,
        baseline_metadata: Sequence[SwarmMeta],
        current_metadata: Sequence[SwarmMeta],
    ) -> Dict[str, Any]:
        """Compute swarm-level drift between two groups of execution traces.

        Returns ``transition_entropy_delta``,
        ``avg_reflection_iterations_delta``, and ``is_swarm_drifted``.
        """
        agents: List[str] = []
        seen: set[str] = set()
        for payload in [*baseline_metadata, *current_metadata]:
            for hop in _agent_hops(payload):
                if hop not in seen:
                    seen.add(hop)
                    agents.append(hop)

        baseline_matrix = _transition_matrix(baseline_metadata, agents)
        current_matrix = _transition_matrix(current_metadata, agents)
        baseline_entropy = _matrix_entropy(baseline_matrix)
        current_entropy = _matrix_entropy(current_matrix)
        transition_entropy_delta = float(current_entropy - baseline_entropy)

        baseline_reflections = np.array(
            [_reflection_iterations(p) for p in baseline_metadata], dtype=float
        )
        current_reflections = np.array(
            [_reflection_iterations(p) for p in current_metadata], dtype=float
        )
        baseline_avg = (
            float(np.mean(baseline_reflections)) if baseline_reflections.size else 0.0
        )
        current_avg = (
            float(np.mean(current_reflections)) if current_reflections.size else 0.0
        )
        reflection_delta = float(current_avg - baseline_avg)

        is_swarm_drifted = (
            abs(transition_entropy_delta) > self.entropy_threshold
            or abs(reflection_delta) > self.reflection_threshold
        )
        if is_swarm_drifted:
            logger.warning(
                f"Swarm drift detected: transition_entropy_delta="
                f"{transition_entropy_delta:.4f} "
                f"(threshold={self.entropy_threshold:.4f}), "
                f"reflection_delta={reflection_delta:.4f} "
                f"(threshold={self.reflection_threshold:.4f})"
            )
        else:
            logger.info(
                f"Swarm behavior stable: transition_entropy_delta="
                f"{transition_entropy_delta:.4f}, "
                f"reflection_delta={reflection_delta:.4f}"
            )

        return {
            "transition_entropy_delta": transition_entropy_delta,
            "avg_reflection_iterations_delta": reflection_delta,
            "is_swarm_drifted": is_swarm_drifted,
        }
