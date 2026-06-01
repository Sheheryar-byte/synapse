import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Manual schema definition
search_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the internet for context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"}
            },
            "required": ["query"]
        }
    }
}

def search_web(query: str, max_results: int = 5) -> str:
    return f"Results for {query}..."

async def main():
    try:
        print("Running OpenAI SDK Example...")
        messages = [{"role": "user", "content": "The history of AI"}]
        
        # Will fail here if OPENAI_API_KEY is not set
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[search_tool_schema]
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"FAILED (As expected without API key): {e}")

if __name__ == "__main__":
    asyncio.run(main())
