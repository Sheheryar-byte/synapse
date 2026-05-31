import asyncio
from typing import TypedDict
import sys

try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, END
except ImportError:
    print("FAILED (As expected): Skipping LangChain example because langchain-openai, langchain-anthropic, and langgraph are not installed.")
    sys.exit(1)

# Boilerplate State
class GraphState(TypedDict):
    topic: str
    notes: str
    final_draft: str

@tool
def search_web(query: str) -> str:
    """Search the internet for context."""
    return f"Results for {query}..."

research_model = ChatOpenAI(model="gpt-4o").bind_tools([search_web])
writer_model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

async def researcher_node(state: GraphState):
    msg = await research_model.ainvoke(state["topic"])
    return {"notes": msg.content}

async def writer_node(state: GraphState):
    msg = await writer_model.ainvoke(state["notes"])
    return {"final_draft": msg.content}

async def main():
    try:
        # Boilerplate Graph
        workflow = StateGraph(GraphState)
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("writer", writer_node)
        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "writer")
        workflow.add_edge("writer", END)
        app = workflow.compile()
        
        result = await app.ainvoke({"topic": "The history of AI"})
        print(result["final_draft"])
    except Exception as e:
        print(f"FAILED (As expected without API keys): {e}")

if __name__ == "__main__":
    asyncio.run(main())
