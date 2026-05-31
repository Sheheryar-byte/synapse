"""
Minimal @workflow decorator and step() helper for Synapse v0.1.

In v0.1 this is a thin convenience wrapper — it simply runs the decorated
coroutine as-is. Checkpointing, retry-on-failure and resumable state are
scheduled for v0.5.
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Optional


async def step(callable_: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a single workflow step.

    In v0.1 this is a passthrough. In v0.5 it will persist a checkpoint to
    Postgres before and after execution, enabling resume-on-failure.

    Usage::

        @workflow
        async def research_pipeline(topic: str):
            data  = await step(researcher, topic)
            draft = await step(writer, data)
            return await step(reviewer, draft)
    """
    raw = callable_(*args, **kwargs)
    if asyncio.iscoroutine(raw):
        return await raw
    return raw


def workflow(fn: Callable) -> Callable:
    """Decorator that marks an async function as a Synapse workflow.

    In v0.1 this simply ensures the function is awaitable and preserves
    metadata. Observability and checkpointing come in v0.5.

    Usage::

        @workflow
        async def my_pipeline(input: str) -> str:
            result = await step(my_agent, input)
            return result
    """
    if not asyncio.iscoroutinefunction(fn):
        raise TypeError(
            f"@workflow requires an async function. '{fn.__name__}' is not async."
        )

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    wrapper.__synapse_workflow__ = True  # type: ignore[attr-defined]
    return wrapper
