import logging
import uuid
from typing import Any, Dict

from pythonjsonlogger.jsonlogger import JsonFormatter

correlation_id = str(uuid.uuid4())


def setup_logging(
    level: int = logging.INFO,
    log_format: str = None,
) -> logging.Logger:
    if log_format is None:
        log_format = (
            "%(asctime)s %(levelname)s %(name)s %(message)s "
            "module=%(module)s function=%(funcName)s "
            "correlation_id=%(correlation_id)s"
        )

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(log_format))

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
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_with_context(
    logger: logging.Logger, level: str, msg: str, **kwargs: Any
) -> None:
    extra: Dict[str, Any] = {"correlation_id": correlation_id}
    extra.update(kwargs)

    log_method = getattr(logger, level, logger.info)
    log_method(msg, extra=extra)
