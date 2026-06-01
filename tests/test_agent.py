"""Tests for the @agent decorator using MockLLM."""
from __future__ import annotations

import pytest

from synapse import agent, tool, AgentError, MockLLM


# ---------------------------------------------------------------------------
# Basic agent call
# ---------------------------------------------------------------------------

@agent(model="gpt-4o")
async def simple_agent(prompt: str) -> str:
    """A minimal agent for testing."""
    ...


class TestAgentBasic:
    @pytest.mark.asyncio
    async def test_agent_returns_mock_response(self):
        with MockLLM("Hello from mock!"):
            result = await simple_agent("test prompt")
        assert result == "Hello from mock!"

    @pytest.mark.asyncio
    async def test_agent_call_count(self):
        with MockLLM("response") as mock:
            await simple_agent("hello")
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_agent_multiple_calls(self):
        with MockLLM("resp") as mock:
            await simple_agent("a")
            await simple_agent("b")
        assert mock.call_count == 2


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

@agent(model="claude-3-5-sonnet")
async def claude_agent(prompt: str) -> str: ...

@agent(model="ollama/llama3")
async def ollama_agent(prompt: str) -> str: ...


class TestModelVariants:
    @pytest.mark.asyncio
    async def test_claude_agent_uses_mock(self):
        with MockLLM("Claude mock response"):
            result = await claude_agent("question")
        assert result == "Claude mock response"

    @pytest.mark.asyncio
    async def test_ollama_agent_uses_mock(self):
        with MockLLM("Ollama mock response"):
            result = await ollama_agent("question")
        assert result == "Ollama mock response"


# ---------------------------------------------------------------------------
# must_not guardrail
# ---------------------------------------------------------------------------

@agent(model="gpt-4o", must_not=["confidential"], max_retries=2)
async def safe_agent(prompt: str) -> str: ...


class TestGuardrail:
    @pytest.mark.asyncio
    async def test_must_not_raises_after_retries(self):
        """When every response contains the forbidden phrase, raise AgentError."""
        with MockLLM("This is confidential information"):
            with pytest.raises(AgentError, match="failed after"):
                await safe_agent("tell me something")

    @pytest.mark.asyncio
    async def test_clean_response_passes_guardrail(self):
        with MockLLM("This is totally safe"):
            result = await safe_agent("tell me something")
        assert result == "This is totally safe"


# ---------------------------------------------------------------------------
# Sync function raises TypeError
# ---------------------------------------------------------------------------

class TestAgentTypeError:
    def test_sync_function_raises(self):
        with pytest.raises(TypeError, match="async function"):
            @agent(model="gpt-4o")
            def sync_fn(prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# >> chaining produces Chain
# ---------------------------------------------------------------------------

class TestAgentChaining:
    def test_rshift_returns_chain(self):
        from synapse.chain import Chain
        chain = simple_agent >> claude_agent
        assert isinstance(chain, Chain)

    @pytest.mark.asyncio
    async def test_chained_agents_execute(self):
        """Each agent in the chain should be called with the previous output."""
        with MockLLM("chained result") as mock:
            pipeline = simple_agent >> claude_agent
            result = await pipeline.run("start")
        # Two agents called: call_count == 2
        assert mock.call_count == 2
        assert result == "chained result"
