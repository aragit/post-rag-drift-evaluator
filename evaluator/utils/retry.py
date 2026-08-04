import logging
from collections.abc import Callable
from typing import Any, TypeVar

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

logger = logging.getLogger("Retry")

F = TypeVar("F", bound=Callable[..., Any])

_RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.Timeout,
)


def _retry_litellm(func: F) -> F:
    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        stop=(stop_after_attempt(3) | stop_after_delay(30)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def call_with_retry(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a litellm function with retry logic for transient API errors."""
    return _retry_litellm(func)(*args, **kwargs)


async def async_call_with_retry(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Call an *async* litellm function with retry logic for transient API errors.

    Uses tenacity's async retry semantics so ``litellm.acompletion`` /
    ``litellm.aembedding`` never block the event loop across retries.
    """

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3) | stop_after_delay(30),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _async_wrapper() -> Any:
        return await func(*args, **kwargs)

    return await _async_wrapper()
