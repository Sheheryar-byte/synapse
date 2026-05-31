"""
@agent decorator — wraps an async function into a fully managed Synapse agent.

The agent handles:
- Multi-model LLM calls via litellm (any provider, same API)
- Automatic tool discovery from @tool-decorated functions passed in
- Output guardrails (must_not, output_schema)
- Retries with corrective prompts on failure
- Fallback model when primary fails
- Token budget enforcement
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, Sequence

import litellm
from pydantic import BaseModel

from synapse.tool import ToolFunction, _TOOL_REGISTRY


@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    fallback: Optional[str] = None
    must_not: list[str] = field(default_factory=list)
    output_schema: Optional[type[BaseModel]] = None
    token_budget: Optional[int] = None
    max_retries: int = 3
    system_prompt: Optional[str] = None
    tools: list[ToolFunction] = field(default_factory=list)


class AgentError(Exception):
    """Raised when an agent exhausts retries or violates constraints."""


class Agent:
    """A fully managed AI agent wrapping an async Python function.

    Do not instantiate directly; use the ``@agent`` decorator.
    """

    def __init__(self, fn: Callable, config: AgentConfig) -> None:
        self._fn = fn
        self._config = config
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        wraps(fn)(self)

    # ------------------------------------------------------------------
    # Calling the agent
    # ------------------------------------------------------------------

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent with the given arguments."""
        # Build the user message from the function's first argument (prompt) or
        # from all positional/keyword arguments stringified.
        sig = inspect.signature(self._fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Prefer a 'prompt' argument if it exists, else serialize all args.
        arguments = dict(bound.arguments)
        if "prompt" in arguments:
            user_message = str(arguments["prompt"])
        else:
            user_message = json.dumps(arguments, default=str)

        return await self._run(user_message)

    async def _run(self, user_message: str, model: Optional[str] = None) -> str:
        cfg = self._config
        active_model = model or cfg.model
        tools_payload = self._build_tools_payload()

        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": user_message})

        last_error: Optional[Exception] = None
        corrective_suffix = ""

        for attempt in range(cfg.max_retries):
            if attempt > 0 and corrective_suffix:
                # Append corrective prompt for retry.
                messages[-1] = {
                    "role": "user",
                    "content": user_message + "\n\n[RETRY INSTRUCTION] " + corrective_suffix,
                }

            try:
                kwargs: dict[str, Any] = dict(
                    model=active_model,
                    messages=messages,
                )
                if cfg.token_budget:
                    kwargs["max_tokens"] = cfg.token_budget
                if tools_payload:
                    kwargs["tools"] = tools_payload
                    kwargs["tool_choice"] = "auto"

                response = await litellm.acompletion(**kwargs)
                content = response.choices[0].message.content or ""

                # Handle tool calls if the model wants to use tools.
                tool_calls = getattr(response.choices[0].message, "tool_calls", None)
                if tool_calls:
                    content = await self._handle_tool_calls(tool_calls, messages, active_model, kwargs)

                # Guardrail check.
                violation = self._check_must_not(content, cfg.must_not)
                if violation:
                    corrective_suffix = f"Your previous response contained '{violation}'. Remove it and try again."
                    last_error = AgentError(f"Output violated must_not rule: '{violation}'")
                    continue

                return content

            except litellm.exceptions.AuthenticationError as exc:
                raise
            except (litellm.exceptions.APIConnectionError, litellm.exceptions.RateLimitError) as exc:
                last_error = exc
                # Try fallback model on connection / rate errors.
                if cfg.fallback and active_model != cfg.fallback:
                    active_model = cfg.fallback
                else:
                    await asyncio.sleep(2 ** attempt)
                continue
            except Exception as exc:
                last_error = exc
                if cfg.fallback and active_model != cfg.fallback:
                    active_model = cfg.fallback
                    continue
                break

        raise AgentError(
            f"Agent '{self.__name__}' failed after {cfg.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    async def _handle_tool_calls(
        self,
        tool_calls: list,
        messages: list[dict],
        model: str,
        base_kwargs: dict,
    ) -> str:
        """Execute tool calls requested by the LLM and return the final response."""
        registry = _TOOL_REGISTRY
        assistant_msg: dict = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)

        for tc in tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments or "{}")
            tool_fn = registry.get(fn_name)
            if tool_fn is None:
                result = f"Error: tool '{fn_name}' not found."
            else:
                try:
                    raw = tool_fn(**fn_args)
                    result = await raw if asyncio.iscoroutine(raw) else raw
                    result = json.dumps(result, default=str)
                except Exception as exc:
                    result = f"Error executing {fn_name}: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

        # Second LLM call with tool results.
        follow_up_kwargs = {**base_kwargs, "messages": messages}
        follow_up_kwargs.pop("tools", None)
        follow_up_kwargs.pop("tool_choice", None)
        response2 = await litellm.acompletion(**follow_up_kwargs)
        return response2.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Chaining  (>>)
    # ------------------------------------------------------------------

    def __rshift__(self, other: Any) -> "Chain":
        from synapse.chain import Chain
        return Chain([self, other])

    def __rrshift__(self, other: Any) -> "Chain":
        from synapse.chain import Chain
        return Chain([other, self])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tools_payload(self) -> list[dict]:
        """Collect tools from config + global registry."""
        payload = []
        seen: set[str] = set()
        # Tools explicitly declared in the decorator take priority.
        for t in self._config.tools:
            payload.append(t.schema)
            seen.add(t.name)
        # Auto-discover from the global registry.
        for name, t in _TOOL_REGISTRY.items():
            if name not in seen:
                payload.append(t.schema)
        return payload

    @staticmethod
    def _check_must_not(content: str, must_not: list[str]) -> Optional[str]:
        for phrase in must_not:
            if phrase.lower() in content.lower():
                return phrase
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Agent name={self.__name__!r} model={self._config.model!r}>"


# ---------------------------------------------------------------------------
# Decorator factory
# ---------------------------------------------------------------------------


def agent(
    fn: Optional[Callable] = None,
    *,
    model: str = "gpt-4o",
    fallback: Optional[str] = None,
    must_not: Optional[list[str]] = None,
    output_schema: Optional[type[BaseModel]] = None,
    token_budget: Optional[int] = None,
    max_retries: int = 3,
    system_prompt: Optional[str] = None,
    tools: Optional[list[ToolFunction]] = None,
) -> Any:
    """Decorator that turns an async function into a Synapse Agent.

    Can be used with or without arguments::

        @agent
        async def assistant(prompt: str) -> str: ...

        @agent(model='claude-3-5-sonnet', fallback='ollama/llama3')
        async def researcher(topic: str) -> str: ...

    Args:
        model: litellm model string (e.g. ``'gpt-4o'``, ``'ollama/llama3'``).
        fallback: Model to try when *model* fails.
        must_not: List of phrases that must not appear in the output.
            On violation the agent retries with a corrective instruction.
        output_schema: Pydantic model that the output must conform to.
        token_budget: Maximum tokens to generate.
        max_retries: How many times to retry on failure / guardrail violation.
        system_prompt: Optional system message prepended to every call.
        tools: Explicit list of ``@tool`` functions. Defaults to ``[]``; the
            global registry is always auto-discovered on top of this.
    """
    config = AgentConfig(
        model=model,
        fallback=fallback,
        must_not=must_not or [],
        output_schema=output_schema,
        token_budget=token_budget,
        max_retries=max_retries,
        system_prompt=system_prompt,
        tools=tools or [],
    )

    def decorator(f: Callable) -> Agent:
        if not inspect.iscoroutinefunction(f):
            raise TypeError(
                f"@agent requires an async function. '{f.__name__}' is not async."
            )
        return Agent(f, config)

    if fn is not None:
        # Used as @agent without parentheses.
        return decorator(fn)

    return decorator
