"""
@tool decorator — auto-generates OpenAI-compatible JSON schema from
a Python function's type annotations and docstring.
"""
from __future__ import annotations

import inspect
import typing
from functools import wraps
from typing import Any, Callable, get_type_hints

from pydantic import TypeAdapter
from pydantic.json_schema import JsonSchemaValue

# Registry of all registered tools in the current process.
# Used for schema inspection and CLI introspection (e.g. `synapse tools list`).
# NOTE: this registry is intentionally NOT injected into agent LLM calls.
# Each agent only receives the tools it explicitly declares via tools=[...].
# Auto-injecting the global registry caused "tool bleed" where agents with no
# tools=[...] would receive schemas they never declared, causing spurious
# tool calls and validation errors on models like Groq.
_TOOL_REGISTRY: dict[str, "ToolFunction"] = {}


def _python_type_to_json_schema(annotation: Any) -> JsonSchemaValue:
    """Use pydantic TypeAdapter to convert a Python type to a JSON schema dict."""
    try:
        return TypeAdapter(annotation).json_schema()
    except Exception:
        return {"type": "string"}


def _build_schema(fn: Callable) -> dict:
    """Build an OpenAI function-calling schema for *fn*."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""

    properties: dict[str, JsonSchemaValue] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = hints.get(name, Any)
        prop_schema = _python_type_to_json_schema(annotation)

        # Pull inline description from docstring (simple heuristic: look for
        # ":param name: …" or "name: …" lines).
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{name}:") or stripped.startswith(f":param {name}:"):
                desc = stripped.split(":", 1)[-1].strip()
                prop_schema["description"] = desc
                break

        properties[name] = prop_schema

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": doc.splitlines()[0] if doc else "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class ToolFunction:
    """Wrapper returned by the @tool decorator.

    Behaves like the original function but also carries a ``.schema`` attribute
    containing the OpenAI-compatible tool definition, and a ``.name`` attribute.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self.name: str = fn.__name__
        self.schema: dict = _build_schema(fn)
        wraps(fn)(self)
        # Preserve introspection attributes.
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__annotations__ = getattr(fn, "__annotations__", {})

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    def __await__(self):
        # If the underlying function is a coroutine function, calling it returns
        # a coroutine; expose __await__ so ``await tool(...)`` works too.
        result = self._fn
        if inspect.iscoroutinefunction(result):
            raise TypeError(
                f"ToolFunction '{self.name}' wraps a coroutine — call it and then await: "
                f"`await {self.name}(...)`"
            )
        raise TypeError(f"ToolFunction '{self.name}' is not awaitable. Call it directly.")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ToolFunction name={self.name!r}>"


def tool(fn: Callable) -> ToolFunction:
    """Decorator that turns a plain function into a Synapse tool.

    Usage::

        @tool
        def search_web(query: str, max_results: int = 5) -> list[str]:
            \"\"\"Search the web. Returns top results.\"\"\"
            ...

    The decorated function gains a ``.schema`` attribute containing the
    OpenAI-compatible JSON tool definition, automatically inferred from the
    type annotations and docstring.
    """
    wrapped = ToolFunction(fn)
    _TOOL_REGISTRY[fn.__name__] = wrapped
    return wrapped
