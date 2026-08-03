from __future__ import annotations

import os

from fastapi import FastAPI

from api.routes.attribution import router as attribution_router
from api.routes.counterfactual import router as counterfactual_router
from api.routes.drift import router as drift_router
from api.routes.optimization import router as optimization_router


def create_app(store_path: str | None = None) -> FastAPI:
    """Create the Sentrix Evaluator API application.

    Args:
        store_path: Path to the JSONL history file.  When ``None``,
            falls back to the ``EVALUATOR_HISTORY_PATH`` environment variable.

    Returns:
        A configured :class:`FastAPI` instance with all Phase 8 routers.
    """
    resolved_path = store_path or os.environ.get(
        "EVALUATOR_HISTORY_PATH", "/tmp/opencode/evaluator_history.jsonl"
    )

    app = FastAPI(
        title="Sentrix Evaluator API",
        version="0.1.0",
        description=(
            "API for drift detection, causal attribution, counterfactual "
            "simulation, and optimization recommendations."
        ),
    )

    app.state.store_path = resolved_path

    app.include_router(drift_router)
    app.include_router(attribution_router)
    app.include_router(counterfactual_router)
    app.include_router(optimization_router)

    return app


app = create_app()
