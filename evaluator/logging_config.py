import contextvars
import logging
import uuid
from typing import Any

from pythonjsonlogger.jsonlogger import JsonFormatter

_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def set_correlation_id(cid: str | None = None) -> str:
    """Bind a correlation id to the current async/task context.

    Generates a fresh v4 hex id when none is supplied.  The id is stored in a
    :class:`contextvars.ContextVar` so each concurrent request/task carries its
    own value without thread-global leakage.
    """
    value = cid or uuid.uuid4().hex
    _correlation_id.set(value)
    return value


def get_correlation_id() -> str:
    """Return the correlation id for the current context (``""`` if unset)."""
    return _correlation_id.get()


class CorrelationIdFilter(logging.Filter):
    """Inject the current correlation id into every emitted log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id"):
            record.correlation_id = get_correlation_id()
        return True


def setup_logging(
    level: int = logging.INFO,
    log_format: str | None = None,
) -> logging.Logger:
    if log_format is None:
        log_format = (
            "%(asctime)s %(levelname)s %(name)s %(message)s "
            "module=%(module)s function=%(funcName)s "
            "correlation_id=%(correlation_id)s"
        )

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(log_format))
    handler.addFilter(CorrelationIdFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler.addFilter(CorrelationIdFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_with_context(
    logger: logging.Logger, level: str, msg: str, **kwargs: Any
) -> None:
    extra: dict[str, Any] = {"correlation_id": get_correlation_id()}
    extra.update(kwargs)

    log_method = getattr(logger, level, logger.info)
    log_method(msg, extra=extra)
