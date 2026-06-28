from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


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
