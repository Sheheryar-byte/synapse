import asyncio
from synapse import agent, tool, MockLLM

# 1. Provide tools
@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the internet for context."""
    return f"Results for {query}..."

# 2. Define agents
@agent(model='gpt-4o', tools=[search_web])
async def researcher(topic: str) -> str: ...

@agent(model='claude-3-5-sonnet')
async def writer(notes: str) -> str: ...

# 3. Chain them together effortlessly
async def main():
    pipeline = researcher >> writer
    
    # We use MockLLM to test this immediately without needing API keys
    with MockLLM(response="This is the final draft about the History of AI."):
        result = await pipeline("The history of AI")
        print("====== SYNAPSE PIPELINE OUTPUT ======")
        print(result)
        print("=====================================")

if __name__ == "__main__":
    asyncio.run(main())
