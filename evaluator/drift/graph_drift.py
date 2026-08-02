"""Topology-level drift detection for GraphRAG outputs.

Compares the structural properties of the sub-graphs emitted by a
GraphRAG system between a baseline and a current window:

- **Density shift**: ``E / (V * (V - 1))`` for each pooled graph.
- **Spectral distance**: Euclidean distance between the top-k eigenvalues
  of the normalized Laplacians, computed with ``scipy.linalg.eigh``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.linalg import eigh

from evaluator.logging_config import get_logger
from evaluator.schemas.telemetry import GraphTopologyPayload

logger = get_logger("GraphDriftCalculator")

GraphLike = GraphTopologyPayload | dict[str, Any]


def _node_list(payload: GraphLike) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return payload.get("nodes", []) or []
    return payload.nodes or []


def _edge_list(payload: GraphLike) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return payload.get("edges", []) or []
    return payload.edges or []


def _edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    source = edge.get("source")
    target = edge.get("target")
    if source is None:
        source = edge.get("from")
    if target is None:
        target = edge.get("to")
    return str(source), str(target)


def _graph_density(num_nodes: int, num_edges: int) -> float:
    if num_nodes < 2:
        return 0.0
    return num_edges / (num_nodes * (num_nodes - 1))


def _pooled_graph(
    payloads: Sequence[GraphLike],
) -> tuple[int, int, np.ndarray]:
    """Merge a group of graph payloads into a single pooled graph."""
    nodes: list[str] = []
    seen: set[str] = set()
    edge_pairs: set[tuple[str, str]] = set()

    for payload in payloads:
        for node in _node_list(payload):
            node_id = node.get("id")
            if node_id is None:
                node_id = str(node)
            node_id = str(node_id)
            if node_id not in seen:
                seen.add(node_id)
                nodes.append(node_id)
        for edge in _edge_list(payload):
            source, target = _edge_endpoints(edge)
            if source and target and source != target:
                edge_pairs.add((source, target))

    index = {node_id: i for i, node_id in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
    for source, target in edge_pairs:
        if source in index and target in index:
            adjacency[index[source], index[target]] = 1.0
            adjacency[index[target], index[source]] = 1.0

    return len(nodes), len(edge_pairs), adjacency


def _normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    if n == 1:
        return np.zeros((1, 1))
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inverse_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    return np.eye(n) - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


def _spectral_distance(
    adjacency_b: np.ndarray, adjacency_c: np.ndarray, k: int
) -> float:
    if adjacency_b.shape[0] == 0 or adjacency_c.shape[0] == 0:
        return 0.0
    eigen_b = eigh(_normalized_laplacian(adjacency_b), eigvals_only=True)
    eigen_c = eigh(_normalized_laplacian(adjacency_c), eigvals_only=True)
    k = max(1, min(k, len(eigen_b), len(eigen_c)))
    return float(np.linalg.norm(eigen_b[-k:] - eigen_c[-k:]))


class GraphDriftCalculator:
    """Compare GraphRAG sub-graph topologies across evaluation windows."""

    def __init__(
        self,
        spectral_threshold: float = 0.5,
        density_threshold: float = 0.1,
        spectral_k: int = 5,
    ):
        self.spectral_threshold = spectral_threshold
        self.density_threshold = density_threshold
        self.spectral_k = spectral_k

    def compute_graph_drift(
        self,
        baseline_graphs: Sequence[GraphLike],
        current_graphs: Sequence[GraphLike],
    ) -> dict[str, Any]:
        """Compute graph-topology drift between two groups of sub-graphs.

        Returns ``spectral_distance``, ``density_delta``,
        ``node_count_delta``, and ``is_graph_drifted``.
        """
        baseline_nodes, baseline_edges, baseline_adj = _pooled_graph(baseline_graphs)
        current_nodes, current_edges, current_adj = _pooled_graph(current_graphs)

        baseline_density = _graph_density(baseline_nodes, baseline_edges)
        current_density = _graph_density(current_nodes, current_edges)
        density_delta = float(current_density - baseline_density)
        node_count_delta = int(current_nodes - baseline_nodes)

        spectral_distance = _spectral_distance(
            baseline_adj, current_adj, self.spectral_k
        )

        is_graph_drifted = (
            spectral_distance > self.spectral_threshold
            or abs(density_delta) > self.density_threshold
        )
        if is_graph_drifted:
            logger.warning(
                f"Graph drift detected: spectral_distance={spectral_distance:.4f} "
                f"(threshold={self.spectral_threshold:.4f}), "
                f"density_delta={density_delta:.4f} "
                f"(threshold={self.density_threshold:.4f})"
            )
        else:
            logger.info(
                f"Graph topology stable: spectral_distance={spectral_distance:.4f}, "
                f"density_delta={density_delta:.4f}"
            )

        return {
            "spectral_distance": spectral_distance,
            "density_delta": density_delta,
            "node_count_delta": node_count_delta,
            "is_graph_drifted": is_graph_drifted,
        }
