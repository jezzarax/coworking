from __future__ import annotations

import asyncio
import functools
import logging
import os
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

OTEL_EXPORTER_OTLP_ENDPOINT_ENVVAR = "OTEL_EXPORTER_OTLP_ENDPOINT"


def make_sync[**P, R](
    func: Callable[P, Awaitable[R]],
) -> Callable[P, R]:
    """Wrap an async function so it can be called from a sync context.

    Each call gets its own ``asyncio.Runner`` so we never share a loop
    across invocations. Mirrors the duckwatcher pattern.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with asyncio.Runner() as runner:
            return runner.run(func(*args, **kwargs))

    return wrapper


def init_trace_recording(otlp_endpoint: str | None):

    env_otlp_endpoint = os.getenv(OTEL_EXPORTER_OTLP_ENDPOINT_ENVVAR, "")
    is_env_otlp_endpoint_set = len(env_otlp_endpoint) > 0

    if otlp_endpoint and is_env_otlp_endpoint_set and otlp_endpoint != is_env_otlp_endpoint_set:
        logger.warning(
            "OTLP endpoint passed via envvar is different than the one passed via command line. Forcing the one passed via command line"
        )
        os.environ[OTEL_EXPORTER_OTLP_ENDPOINT_ENVVAR] = otlp_endpoint
        is_env_otlp_endpoint_set = True

    if not is_env_otlp_endpoint_set:
        raise RuntimeError(
            f"{OTEL_EXPORTER_OTLP_ENDPOINT_ENVVAR} is not set. Tracing requires an OTLP endpoint."
        )

    import logfire

    logfire.configure(send_to_logfire=False)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)
    logfire.instrument_openai()
