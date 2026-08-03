from __future__ import annotations

from fastapi import HTTPException, Request

from evaluator.storage import JSONHistoryStore


def get_store(request: Request) -> JSONHistoryStore:
    """FastAPI dependency that provides a :class:`JSONHistoryStore`.

    The store path is read from ``app.state.store_path``, which can be
    set at app-creation time or overridden per-request in tests.
    """
    path = getattr(request.app.state, "store_path", None)
    if path is None:
        raise HTTPException(
            status_code=500,
            detail="History store path is not configured. Set store_path in app state.",
        )
    return JSONHistoryStore(path)
