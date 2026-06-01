"""Tests for the @tool decorator and ToolFunction."""
from __future__ import annotations

import pytest

from synapse.tool import tool, ToolFunction, _build_schema


# ---------------------------------------------------------------------------
# Simple tools used across tests
# ---------------------------------------------------------------------------

@tool
def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web. Returns top results."""
    return [f"result for {query}"] * max_results


@tool
def no_args() -> str:
    """A tool with no arguments."""
    return "ok"


@tool
def required_and_optional(name: str, age: int = 18) -> str:
    """A tool with one required and one optional param."""
    return f"{name},{age}"


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

class TestSchemaGeneration:
    def test_schema_has_function_key(self):
        assert "function" in search_web.schema

    def test_schema_name_matches_function(self):
        assert search_web.schema["function"]["name"] == "search_web"

    def test_schema_description_from_docstring(self):
        desc = search_web.schema["function"]["description"]
        assert "Search the web" in desc

    def test_schema_has_parameters(self):
        params = search_web.schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "max_results" in params["properties"]

    def test_query_type_is_string(self):
        props = search_web.schema["function"]["parameters"]["properties"]
        assert props["query"]["type"] == "string"

    def test_max_results_type_is_integer(self):
        props = search_web.schema["function"]["parameters"]["properties"]
        assert props["max_results"]["type"] == "integer"

    def test_no_args_tool_has_empty_required(self):
        assert no_args.schema["function"]["parameters"]["required"] == []


# ---------------------------------------------------------------------------
# Required vs optional fields
# ---------------------------------------------------------------------------

class TestRequiredFields:
    def test_required_contains_name_not_age(self):
        required = required_and_optional.schema["function"]["parameters"]["required"]
        assert "name" in required
        assert "age" not in required

    def test_search_web_query_is_required(self):
        required = search_web.schema["function"]["parameters"]["required"]
        assert "query" in required

    def test_search_web_max_results_is_optional(self):
        required = search_web.schema["function"]["parameters"]["required"]
        assert "max_results" not in required


# ---------------------------------------------------------------------------
# ToolFunction callable behaviour
# ---------------------------------------------------------------------------

class TestToolCallable:
    def test_tool_is_tool_function_instance(self):
        assert isinstance(search_web, ToolFunction)

    def test_tool_is_directly_callable(self):
        result = search_web("python", max_results=2)
        assert len(result) == 2

    def test_tool_name_attribute(self):
        assert search_web.name == "search_web"

    def test_tool_preserves_docstring(self):
        assert "Search the web" in (search_web.__doc__ or "")


# ---------------------------------------------------------------------------
# Async tool
# ---------------------------------------------------------------------------

@tool
async def async_greet(name: str) -> str:
    """Greet someone asynchronously."""
    return f"Hello, {name}!"


class TestAsyncTool:
    @pytest.mark.asyncio
    async def test_async_tool_returns_coroutine(self):
        coro = async_greet("Alice")
        import asyncio, inspect
        assert inspect.iscoroutine(coro)
        result = await coro
        assert result == "Hello, Alice!"

    def test_async_tool_schema_generated(self):
        assert "name" in async_greet.schema["function"]["parameters"]["properties"]
