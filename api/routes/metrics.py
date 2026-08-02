from fastapi import APIRouter, Response

from api.metrics import get_content_type, render_metrics

router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint exposing all registered metrics."""
    return Response(content=render_metrics(), media_type=get_content_type())
