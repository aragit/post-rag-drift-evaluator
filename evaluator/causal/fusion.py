"""Causal + Latent Drift Fusion Layer.

Maps dual-track latent distance metrics into Causal DAG node prior
probabilities for root-cause attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluator.latent_drift.schemas import LatentDriftResult


@dataclass
class CausalNode:
    """A node in the causal graph representing a pipeline component.

    Attributes:
        node_id: Unique identifier (e.g., "vector_index", "embedding_model").
        node_type: Category of the node ("retrieval", "generation", "shared").
        prior_failure_prob: Prior probability of this node failing/causing drift.
        description: Human-readable description.
        metadata: Additional context.
    """

    node_id: str
    node_type: str
    prior_failure_prob: float = 0.0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalGraph:
    """A simple causal DAG of pipeline components.

    Attributes:
        nodes: List of :class:`CausalNode` objects.
        edges: List of (source_id, target_id) tuples representing causality.
        metadata: Additional graph-level metadata.
    """

    nodes: list[CausalNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> CausalNode | None:
        """Look up a node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "prior_failure_prob": n.prior_failure_prob,
                    "description": n.description,
                    "metadata": n.metadata,
                }
                for n in self.nodes
            ],
            "edges": list(self.edges),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalGraph:
        nodes = [
            CausalNode(
                node_id=n["node_id"],
                node_type=n["node_type"],
                prior_failure_prob=n.get("prior_failure_prob", 0.0),
                description=n.get("description", ""),
                metadata=n.get("metadata", {}),
            )
            for n in data.get("nodes", [])
        ]
        return cls(
            nodes=nodes,
            edges=[tuple(e) for e in data.get("edges", [])],
            metadata=data.get("metadata", {}),
        )


class CausalLatentFusionEngine:
    """Fuses latent drift scores into causal graph node priors.

    Maps dual-track drift signals to pipeline component failure probabilities:

    - ``"retrieval"`` track → retrieval nodes (VectorIndexNode, EmbeddingModelNode)
    - ``"generation"`` track → generation nodes (PromptTemplateNode, LLMRouterNode)

    The mapping function: ``P(failure) = min(1.0, drift_score * sensitivity)``
    where sensitivity scales the drift score into probability space.

    Args:
        sensitivity: Multiplier for converting drift scores to probabilities
            (default 2.0, so a drift score of 0.5 → P=1.0).
        default_nodes: Whether to use a default causal graph (True) or
            require an explicit graph.
    """

    DEFAULT_SENSITIVITY = 2.0

    def __init__(
        self,
        sensitivity: float = DEFAULT_SENSITIVITY,
        default_nodes: bool = True,
    ):
        self.sensitivity = sensitivity
        self.default_nodes = default_nodes

    @staticmethod
    def _compute_failure_prob(drift_score: float, sensitivity: float) -> float:
        """Compute failure probability from drift score.

        ``P = min(1.0, drift_score * sensitivity)``
        """
        return min(1.0, max(0.0, drift_score * sensitivity))

    def _default_graph(self) -> CausalGraph:
        """Build a default causal graph for RAG pipelines."""
        nodes = [
            CausalNode(
                node_id="vector_index",
                node_type="retrieval",
                prior_failure_prob=0.0,
                description="Vector database / index quality",
            ),
            CausalNode(
                node_id="embedding_model",
                node_type="retrieval",
                prior_failure_prob=0.0,
                description="Embedding model used for retrieval",
            ),
            CausalNode(
                node_id="prompt_template",
                node_type="generation",
                prior_failure_prob=0.0,
                description="Prompt template for LLM generation",
            ),
            CausalNode(
                node_id="llm_router",
                node_type="generation",
                prior_failure_prob=0.0,
                description="LLM routing and model selection",
            ),
        ]
        edges = [
            ("vector_index", "embedding_model"),
            ("embedding_model", "prompt_template"),
            ("prompt_template", "llm_router"),
        ]
        return CausalGraph(nodes=nodes, edges=edges)

    def _resolve_track_score(self, drift_result: LatentDriftResult) -> dict[str, float]:
        """Extract per-track scores from a drift result.

        Uses the metric_breakdown if available, otherwise falls back
        to the main drift_score mapped to the result's track.
        """
        breakdown = drift_result.metric_breakdown
        if breakdown:
            return dict(breakdown)
        return {drift_result.track: drift_result.drift_score}

    def fuse_drift_into_causal_graph(
        self,
        drift_result: LatentDriftResult,
        graph: CausalGraph | None = None,
    ) -> CausalGraph:
        """Map latent drift scores to causal graph node prior probabilities.

        Args:
            drift_result: The latent drift computation result, which may
                include a ``metric_breakdown`` with per-track scores.
            graph: An optional :class:`CausalGraph` to update.  If None,
                a default graph is created.

        Returns:
            An updated :class:`CausalGraph` with prior failure
            probabilities set on nodes based on the drift scores.
        """
        if graph is None:
            if self.default_nodes:
                graph = self._default_graph()
            else:
                graph = CausalGraph()

        # Work on a copy
        updated_nodes = [
            CausalNode(
                node_id=n.node_id,
                node_type=n.node_type,
                prior_failure_prob=n.prior_failure_prob,
                description=n.description,
                metadata=dict(n.metadata),
            )
            for n in graph.nodes
        ]
        nodes_by_type: dict[str, list[CausalNode]] = {}
        for node in updated_nodes:
            nodes_by_type.setdefault(node.node_type, []).append(node)

        track_scores = self._resolve_track_score(drift_result)

        for track, score in track_scores.items():
            if track == "retrieval":
                target_type = "retrieval"
            elif track == "generation":
                target_type = "generation"
            else:
                target_type = track

            nodes = nodes_by_type.get(target_type, [])
            if not nodes:
                continue

            prob = self._compute_failure_prob(score, self.sensitivity)
            for node in nodes:
                # Only update if this is a higher probability
                if prob > node.prior_failure_prob:
                    node.prior_failure_prob = round(prob, 6)

        return CausalGraph(
            nodes=updated_nodes,
            edges=list(graph.edges),
            metadata={
                **dict(graph.metadata),
                "fusion_source": "latent_drift",
                "track_scores": track_scores,
                "sensitivity": self.sensitivity,
            },
        )

    def fuse_dual_track(
        self,
        retrieval_result: LatentDriftResult | None,
        generation_result: LatentDriftResult | None,
        graph: CausalGraph | None = None,
    ) -> CausalGraph:
        """Fuse separate retrieval and generation drift results into a graph.

        Args:
            retrieval_result: Drift result for the retrieval track.
            generation_result: Drift result for the generation track.
            graph: Optional causal graph to update.

        Returns:
            Updated :class:`CausalGraph`.
        """
        if graph is None and self.default_nodes:
            graph = self._default_graph()
        elif graph is None:
            graph = CausalGraph()

        updated_nodes = [
            CausalNode(
                node_id=n.node_id,
                node_type=n.node_type,
                prior_failure_prob=n.prior_failure_prob,
                description=n.description,
                metadata=dict(n.metadata),
            )
            for n in graph.nodes
        ]

        nodes_by_type: dict[str, list[CausalNode]] = {}
        for node in updated_nodes:
            nodes_by_type.setdefault(node.node_type, []).append(node)

        if retrieval_result:
            prob = self._compute_failure_prob(
                retrieval_result.drift_score, self.sensitivity
            )
            for node in nodes_by_type.get("retrieval", []):
                if prob > node.prior_failure_prob:
                    node.prior_failure_prob = round(prob, 6)

        if generation_result:
            prob = self._compute_failure_prob(
                generation_result.drift_score, self.sensitivity
            )
            for node in nodes_by_type.get("generation", []):
                if prob > node.prior_failure_prob:
                    node.prior_failure_prob = round(prob, 6)

        return CausalGraph(
            nodes=updated_nodes,
            edges=list(graph.edges),
            metadata={
                **dict(graph.metadata),
                "fusion_source": "dual_track_drift",
                "retrieval_score": retrieval_result.drift_score
                if retrieval_result
                else 0.0,
                "generation_score": generation_result.drift_score
                if generation_result
                else 0.0,
                "sensitivity": self.sensitivity,
            },
        )
