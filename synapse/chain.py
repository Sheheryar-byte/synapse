"""
Chain — pipes multiple agents/callables together with the >> operator.

Usage::

    pipeline = researcher >> writer >> reviewer
    result   = await pipeline.run("AI frameworks")
    # or equivalently:
    result   = await pipeline("AI frameworks")
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable


class Chain:
    """An ordered sequence of async callables (agents or plain async functions).

    Build via the ``>>`` operator::

        chain = step_a >> step_b >> step_c

    Execute via::

        result = await chain.run(initial_input)
        result = await chain(initial_input)   # alias
    """

    def __init__(self, steps: list[Any]) -> None:
        self._steps: list[Any] = steps

    # ------------------------------------------------------------------
    # Extending the chain
    # ------------------------------------------------------------------

    def __rshift__(self, other: Any) -> "Chain":
        """Append another step and return a new Chain."""
        return Chain(self._steps + [other])

    def __rrshift__(self, other: Any) -> "Chain":
        """Prepend a step and return a new Chain."""
        return Chain([other] + self._steps)

    # ------------------------------------------------------------------
    # Running the chain
    # ------------------------------------------------------------------

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the chain left-to-right, piping each output as the next input.

        The first step receives *args* and *kwargs* directly.
        Subsequent steps receive the previous step's output as their sole argument.
        """
        result: Any = None
        first = True
        for step in self._steps:
            if first:
                raw = step(*args, **kwargs)
                first = False
            else:
                raw = step(result)

            if asyncio.iscoroutine(raw):
                result = await raw
            else:
                result = raw

        return result

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for ``run``."""
        return await self.run(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        names = " >> ".join(getattr(s, "__name__", repr(s)) for s in self._steps)
        return f"<Chain [{names}]>"


def chain(fn: Callable) -> Chain:
    """Wrap a single callable in a :class:`Chain` so that ``>>`` works naturally.

    Example::

        pipeline = chain(upper) >> chain(exclaim)
        result   = await pipeline.run("hello")
    """
    return Chain([fn])
