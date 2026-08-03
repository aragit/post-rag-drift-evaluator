"""Production API gateway for Sentrix Evaluator.

Exposes FastAPI endpoints for evaluation ingestion, drift detection,
and closed-loop remediation.  All operations run in-memory using
:class:`InMemoryHistoryStore` for simulation phases.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from evaluator.counterfactual.simulator import run_counterfactual_analysis
from evaluator.guardrails.policy import PolicyEvaluator
from evaluator.latent_drift import EmbeddingBatch
from evaluator.optimization.engine import OptimizationEngine
from evaluator.optimization.runner import (
    STATUS_APPROVED,
    STATUS_BLOCKED_BY_GUARDRAIL,
    STATUS_NO_ACTION_NEEDED,
    OptimizationRunner,
)
from evaluator.storage import InMemoryHistoryStore, JSONHistoryStore


class EvalRequest(BaseModel):
    """Batch of evaluation records to ingest."""

    records: list[dict[str, Any]] = Field(default_factory=list)


class EvalResponse(BaseModel):
    """Response confirming ingested records."""

    ingested: int = 0
    status: str = "ok"


class DriftDetectRequest(BaseModel):
    """Request to detect drift between baseline and current batches."""

    baseline_vectors: list[list[float]] = Field(default_factory=list)
    current_vectors: list[list[float]] = Field(default_factory=list)
    threshold: float = Field(default=0.15, ge=0.0)
    metric: str = Field(default="mmd")
    pca_components: int = Field(default=5, ge=1)
    track: str = Field(default="unified")


class DriftDetectResponse(BaseModel):
    """Response with latent drift detection results."""

    drift_score: float = 0.0
    drift_detected: bool = False
    metric_used: str = "mmd"
    track: str = "unified"
    n_samples_baseline: int = 0
    n_samples_current: int = 0


class RemediateRequest(BaseModel):
    """Request to run the full closed-loop remediation cycle."""

    drift_event: dict[str, Any]
    attribution: dict[str, Any]
    execution_history: list[dict[str, Any]] = Field(default_factory=list)
    store_records: list[dict[str, Any]] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1)


class RemediateResponse(BaseModel):
    """Response with remediation status and action details."""

    status: str = STATUS_NO_ACTION_NEEDED
    action: dict[str, Any] | None = None
    policy_decision: dict[str, Any] | None = None
    counterfactual_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


def create_production_app(
    store_path: str | None = None,
    min_confidence: float = 0.70,
) -> FastAPI:
    """Build the production FastAPI application.

    Args:
        store_path: Path for the JSONL history store. If None, uses
            a temp file path.
        min_confidence: Minimum counterfactual confidence for action selection.
    """
    import tempfile

    if store_path is None:
        fd, store_path = tempfile.mkstemp(suffix=".jsonl")
        import os
        os.close(fd)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield

    application = FastAPI(
        title="Sentrix Evaluator — Production Gateway",
        version="3.0.0",
        lifespan=lifespan,
    )

    application.state.store_path = store_path
    application.state.store = JSONHistoryStore(store_path)
    application.state.optimization_engine = OptimizationEngine(min_confidence=min_confidence)
    application.state.policy_evaluator = PolicyEvaluator()
    application.state.runner = OptimizationRunner(
        engine=application.state.optimization_engine,
        policy_evaluator=application.state.policy_evaluator,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.post("/v1/eval", response_model=EvalResponse)
    async def evaluate(request: EvalRequest):
        """Ingest evaluation records into the history store."""
        store = application.state.store
        count = 0
        for record_data in request.records:
            record = _dict_to_record(record_data)
            store.save(record)
            count += 1
        return EvalResponse(ingested=count, status="ok")

    @application.post("/v1/drift/detect", response_model=DriftDetectResponse)
    async def detect_drift(request: DriftDetectRequest):
        """Detect dual-track latent drift between baseline and current embeddings."""
        from evaluator.latent_drift import compute_latent_drift

        baseline = EmbeddingBatch(
            vectors=np_array(request.baseline_vectors),
            track=request.track,
        )
        current = EmbeddingBatch(
            vectors=np_array(request.current_vectors),
            track=request.track,
        )

        from evaluator.config import config
        pca_components = request.pca_components or config.PCA_COMPONENTS

        result = compute_latent_drift(
            baseline=baseline,
            current=current,
            threshold=request.threshold,
            metric=request.metric,
            pca_components=pca_components,
        )

        return DriftDetectResponse(
            drift_score=result.drift_score,
            drift_detected=result.drift_detected,
            metric_used=result.metric_used,
            track=result.track,
            n_samples_baseline=result.n_samples_baseline,
            n_samples_current=result.n_samples_current,
        )

    @application.post("/v1/remediate", response_model=RemediateResponse)
    async def remediate(request: RemediateRequest):
        """Run the full closed-loop optimization cycle and return the result."""
        from evaluator.causal.models import CausalAttribution
        from evaluator.temporal.models import DriftEvent

        drift_event = DriftEvent.from_dict(request.drift_event)
        attribution = CausalAttribution.from_dict(request.attribution)

        # Build in-memory store from provided records
        from evaluator.storage import EvaluationRecord
        in_memory = InMemoryHistoryStore()
        for rec_data in request.store_records:
            in_memory.append(EvaluationRecord.from_dict(rec_data))

        # Reconstruct execution history as OptimizationAction objects
        from evaluator.optimization.models import OptimizationAction
        exec_history = [
            OptimizationAction.from_dict(h) for h in request.execution_history
        ]

        # Run counterfactual analysis
        cf_results = run_counterfactual_analysis(
            drift_event=drift_event,
            attribution=attribution,
            store=in_memory,
            top_k=request.top_k,
        )

        # Select action based on drift track
        action = application.state.optimization_engine.select_action(
            drift_event, cf_results, attribution.factors
        )

        if action is None:
            return RemediateResponse(
                status=STATUS_NO_ACTION_NEEDED,
                counterfactual_count=len(cf_results),
            )

        # Validate with policy
        policy_decision = application.state.policy_evaluator.validate_action(
            action, exec_history
        )

        if not policy_decision.allowed:
            result_action = OptimizationAction(
                action_type=action.action_type,
                target_run_id=action.target_run_id,
                change_id=action.change_id,
                description=action.description,
                metadata={
                    **action.metadata,
                    "status": STATUS_BLOCKED_BY_GUARDRAIL,
                    "guardrail_reason": policy_decision.reason,
                    "rule_violated": policy_decision.rule_violated,
                },
            )
            return RemediateResponse(
                status=STATUS_BLOCKED_BY_GUARDRAIL,
                action=result_action.to_dict(),
                policy_decision={
                    "allowed": False,
                    "reason": policy_decision.reason,
                    "rule_violated": policy_decision.rule_violated,
                },
                counterfactual_count=len(cf_results),
            )

        # Approved
        approved_action = OptimizationAction(
            action_type=action.action_type,
            target_run_id=action.target_run_id,
            change_id=action.change_id,
            description=action.description,
            metadata={
                **action.metadata,
                "status": STATUS_APPROVED,
                "executed_at": __import__("time").time(),
            },
        )
        return RemediateResponse(
            status=STATUS_APPROVED,
            action=approved_action.to_dict(),
            policy_decision={
                "allowed": True,
                "reason": policy_decision.reason,
                "rule_violated": None,
            },
            counterfactual_count=len(cf_results),
        )

    return application


def np_array(vectors: list[list[float]]) -> Any:
    """Convert a list of lists to a numpy array."""
    import numpy as np
    return np.array(vectors)


def _dict_to_record(data: dict[str, Any]) -> Any:
    """Convert a dict to an EvaluationRecord, handling nested metrics."""
    from evaluator.storage.models import EvaluationRecord

    records = EvaluationRecord.from_dict(data)
    return records


app = create_production_app()
