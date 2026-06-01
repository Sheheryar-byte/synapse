# Synapse — The AI Application Runtime

> `pip install synapse-runtime`

Not a framework. Not a wrapper. The missing AI layer for your Python stack.

## 30-Second Demo

```python
from synapse import agent, tool, workflow, step

@tool
def search_web(query: str) -> list[str]:
    """Search the web. Returns top results."""
    ...  # your implementation

@agent(model='gpt-4o', fallback='ollama/llama3')
async def researcher(topic: str) -> str: ...

@agent(model='claude-3-5-sonnet')
async def writer(content: str) -> str: ...

# Chain agents with >>
pipeline = researcher >> writer
result   = await pipeline.run('AI application runtimes')
```

That's it. No subclassing. No boilerplate. No JSON schema by hand.

---

## The Four Primitives (v0.1)

### `@tool` — Auto-generated schemas

```python
@tool
def send_email(to: str, subject: str, body: str = '') -> bool:
    """Send an email. Returns True on success."""
    ...
```

Synapse reads the type annotations and docstring. The JSON schema is generated automatically — no `BaseTool`, no `{type: "function", function: {…}}` by hand.

Pass tools explicitly to the agents that need them:

```python
@agent(model='gpt-4o', tools=[search_web])   # only this agent can call search_web
async def researcher(topic: str) -> str: ...

@agent(model='gpt-4o')                        # clean — no tools injected
async def writer(draft: str) -> str: ...
```

Each agent only receives the schemas you declare. No global bleed between agents.

### `@agent` — Any model, same API

```python
@agent(model='gpt-4o')                  # OpenAI
@agent(model='claude-3-5-sonnet')       # Anthropic
@agent(model='gemini/gemini-pro')       # Google
@agent(model='ollama/llama3')           # Local (free)
```

Powered by [litellm](https://github.com/BerriAI/litellm) — 100+ providers, one decorator.

Add guardrails:

```python
@agent(
    model='gpt-4o',
    fallback='ollama/llama3',
    must_not=['PII', 'speculation'],
    token_budget=2000,
)
async def safe_agent(prompt: str) -> str: ...
```

### `>>` — Readable pipelines

```python
pipeline = researcher >> writer >> reviewer
result   = await pipeline.run(topic='climate change')
```

LangGraph needs nodes and edges. Synapse uses `>>`.

### `@workflow` + `step()` — Composable pipelines

```python
@workflow
async def research_pipeline(topic: str) -> str:
    data  = await step(researcher, topic)
    draft = await step(writer, data)
    return await step(reviewer, draft)
```

---

## Zero-Cost Testing

```python
from synapse import MockLLM

def test_my_agent():
    with MockLLM("expected response") as mock:
        result = await my_agent("prompt")
    assert result == "expected response"
    assert mock.call_count == 1
```

`MockLLM` patches litellm. No API calls. No tokens spent.

---

## Synapse vs the alternatives

| Feature | Synapse | LangChain | OpenAI SDK | LangGraph |
|---|---|---|---|---|
| Any LLM provider | ✅ | ✅ | ❌ OpenAI only | ✅ |
| Schema from annotations | ✅ | ❌ manual | ❌ manual | ❌ manual |
| `>>` chaining | ✅ | ❌ verbose | ❌ | ❌ |
| Built-in guardrails | ✅ | ⚠️ external | ❌ | ❌ |
| Readable 3-line agent | ✅ | ❌ subclass | ❌ | ❌ |
| MockLLM for tests | ✅ | ❌ | ❌ | ❌ |

---

## Roadmap

| Version | Name | Ships |
|---|---|---|
| **v0.1** | The Tiny Core | `@agent`, `@tool`, `>>`, MockLLM ← *you are here* |
| v0.3 | Useful Projects | Memory API, streaming, prompt templates, CLI |
| v0.5 | AI Runtime | Workflow engine, checkpointing, observability, guardrails |
| v0.8 | Synapse Identity | Prompt debugger, multi-agent graphs, human-in-the-loop |
| v1.0 | Production | Multi-tenancy, audit trail, intent routing |

---

## Install

```bash
pip install synapse-runtime

# For local LLM (free, no API key needed):
# Install Ollama → https://ollama.ai
ollama pull llama3
```

MIT License · Built in public · [Issues welcome](https://github.com/Sheheryar-byte/synapse/issues)
