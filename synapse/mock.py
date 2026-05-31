"""
MockLLM — deterministic LLM fixture for zero-cost testing.

Usage::

    from synapse.mock import MockLLM

    with MockLLM("Hello from mock!") as mock:
        result = await my_agent("any prompt")
        assert result == "Hello from mock!"
        assert mock.call_count == 1
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch


class MockLLM:
    """Context manager that patches litellm.acompletion to return a fixed response.

    Args:
        response: The text the mock LLM will always return.
        raises: If provided, the mock will raise this exception instead of returning.
    """

    def __init__(self, response: str = "mock response", raises: Exception | None = None) -> None:
        self._response = response
        self._raises = raises
        self.call_count = 0
        self._patches: list[Any] = []

    def _make_mock_response(self, content: str) -> Any:
        """Build an object that mimics the litellm ModelResponse structure."""

        class _Choice:
            class _Message:
                def __init__(self, c: str) -> None:
                    self.content = c
                    self.tool_calls = None
                    self.role = "assistant"

            def __init__(self, c: str) -> None:
                self.message = self._Message(c)
                self.finish_reason = "stop"

        class _Response:
            def __init__(self, c: str) -> None:
                self.choices = [_Choice(c)]
                self.model = "mock"
                self.usage = type("Usage", (), {"total_tokens": len(c.split()), "prompt_tokens": 10, "completion_tokens": len(c.split())})()

        return _Response(content)

    def _side_effect(self, *args: Any, **kwargs: Any):
        self.call_count += 1
        if self._raises:
            raise self._raises
        return self._make_mock_response(self._response)

    def __enter__(self) -> "MockLLM":
        mock_fn = AsyncMock(side_effect=self._side_effect)
        p = patch("litellm.acompletion", mock_fn)
        p.start()
        self._patches.append(p)
        return self

    def __exit__(self, *args: Any) -> None:
        for p in self._patches:
            p.stop()
        self._patches.clear()
