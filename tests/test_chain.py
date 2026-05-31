"""Tests for the Chain / >> operator."""
from __future__ import annotations

import pytest

from synapse import agent, MockLLM
from synapse.chain import Chain, chain as make_chain


# ---------------------------------------------------------------------------
# Simple async functions (not @agent) for chain testing
# ---------------------------------------------------------------------------

async def upper(text: str) -> str:
    return text.upper()

async def exclaim(text: str) -> str:
    return text + "!"

async def repeat(text: str) -> str:
    return text * 2


# ---------------------------------------------------------------------------
# Building chains with Chain([]) for plain callables
# ---------------------------------------------------------------------------

class TestChainBuilding:
    def test_two_steps_returns_chain(self):
        chain = Chain([upper, exclaim])
        assert isinstance(chain, Chain)

    def test_three_steps_returns_chain(self):
        chain = Chain([upper, exclaim, repeat])
        assert isinstance(chain, Chain)
        assert len(chain._steps) == 3

    def test_extending_existing_chain(self):
        chain = Chain([upper, exclaim]) >> repeat
        assert len(chain._steps) == 3

    def test_chain_helper_function(self):
        """chain() helper wraps a callable to support >> with plain functions."""
        pipeline = make_chain(upper) >> make_chain(exclaim)
        assert isinstance(pipeline, Chain)


# ---------------------------------------------------------------------------
# Executing chains
# ---------------------------------------------------------------------------

class TestChainExecution:
    @pytest.mark.asyncio
    async def test_two_step_chain(self):
        pipeline = Chain([upper, exclaim])
        result = await pipeline.run("hello")
        assert result == "HELLO!"

    @pytest.mark.asyncio
    async def test_three_step_chain(self):
        pipeline = Chain([upper, exclaim, repeat])
        result = await pipeline.run("hi")
        assert result == "HI!HI!"

    @pytest.mark.asyncio
    async def test_chain_call_alias(self):
        """Calling the chain directly should work like .run()."""
        pipeline = Chain([upper, exclaim])
        result = await pipeline("world")
        assert result == "WORLD!"

    @pytest.mark.asyncio
    async def test_single_step_chain(self):
        """A chain of one step should work."""
        chain = Chain([upper])
        result = await chain.run("test")
        assert result == "TEST"

    @pytest.mark.asyncio
    async def test_chain_helper_execution(self):
        """make_chain() wraps functions so >> works naturally."""
        pipeline = make_chain(upper) >> make_chain(exclaim)
        result = await pipeline.run("hello")
        assert result == "HELLO!"


# ---------------------------------------------------------------------------
# Agent >> chaining with MockLLM
# ---------------------------------------------------------------------------

@agent(model="gpt-4o")
async def stage_one(prompt: str) -> str: ...

@agent(model="gpt-4o")
async def stage_two(prompt: str) -> str: ...

@agent(model="gpt-4o")
async def stage_three(prompt: str) -> str: ...


class TestAgentChainExecution:
    @pytest.mark.asyncio
    async def test_two_agent_chain(self):
        with MockLLM("pipeline output") as mock:
            result = await (stage_one >> stage_two).run("input")
        assert mock.call_count == 2
        assert result == "pipeline output"

    @pytest.mark.asyncio
    async def test_three_agent_chain(self):
        with MockLLM("final") as mock:
            result = await (stage_one >> stage_two >> stage_three).run("start")
        assert mock.call_count == 3
        assert result == "final"


