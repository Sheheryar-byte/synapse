"""
Synapse — The AI Application Runtime.

Quick start::

    from synapse import agent, tool, workflow, step, MockLLM

    @tool
    def search_web(query: str) -> list[str]:
        \"\"\"Search the web. Returns top results.\"\"\"
        ...

    @agent(model='gpt-4o', fallback='ollama/llama3')
    async def researcher(topic: str) -> str: ...

    @agent(model='claude-3-5-sonnet')
    async def writer(content: str) -> str: ...

    pipeline = researcher >> writer
    result   = await pipeline.run('AI application runtimes')
"""
from synapse.agent import Agent, AgentConfig, AgentError, agent
from synapse.chain import Chain
from synapse.mock import MockLLM
from synapse.tool import ToolFunction, tool
from synapse.workflow import step, workflow

__all__ = [
    # Decorators
    "agent",
    "tool",
    "workflow",
    # Helpers
    "step",
    # Classes
    "Agent",
    "AgentConfig",
    "AgentError",
    "Chain",
    "MockLLM",
    "ToolFunction",
]

__version__ = "0.1.1"
