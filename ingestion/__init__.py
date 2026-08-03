from ingestion.run_schema import RAGRun

__all__ = [
    "AsyncIngestionBuffer",
    "RedisStreamBuffer",
    "RAGRun",
    "get_ingestion_buffer",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies at package init time."""
    if name == "AsyncIngestionBuffer":
        from ingestion.queue import AsyncIngestionBuffer

        return AsyncIngestionBuffer
    if name == "get_ingestion_buffer":
        from ingestion.queue import get_ingestion_buffer

        return get_ingestion_buffer
    if name == "RedisStreamBuffer":
        from ingestion.redis_queue import RedisStreamBuffer

        return RedisStreamBuffer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
