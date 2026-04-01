# ai-playground

Playing with different AI Agent Frameworks with Python.

## Setup

**Prerequisites:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
export GROQ_API_KEY=your_key_here
```

## Examples

### `langchain_minimal_example.py`

A minimal LCEL chain using `ChatPromptTemplate`. Demonstrates:

- Building a prompt with system and user messages, including `{placeholders}`
- Composing a chain with the `|` pipe operator: `prompt | llm | StrOutputParser()`
- Invoking the chain with a dict of template variables

```bash
uv run langchain_minimal_example.py
```

### `langchain_create_agent_example.py`

A CLI chat agent built with `create_agent` from LangChain. Demonstrates:

- Defining tools with `@tool` and returning artifacts using `response_format="content_and_artifact"`
- Building an agent that automatically loops over tool calls until it has enough information to respond
- Streaming the final text response token by token while capturing tool results

```bash
uv run langchain_create_agent_example.py
```

### `langgraph_agent_example.py`

The same agent built manually with LangGraph, exposing the internals that `create_agent` abstracts away. Demonstrates:

- Defining a custom `AppState` with `add_messages` and `tool_results` to accumulate typed artifacts
- Wiring a `StateGraph` with `model`, `tools`, and `collect_artifacts` nodes
- Using `should_continue` as a conditional edge to implement the tool-calling loop
- Streaming text tokens and reading `tool_results` from the final state using dual `stream_mode=["messages", "values"]`

```bash
uv run langgraph_agent_example.py
```

<div align="center">
  <img src="assets/langgraph_agent_example.png" alt="langgraph_agent_example graph" />
</div>

To regenerate:

```bash
GROQ_API_KEY=<GROQ_API_KEY> uv run python -c "
from langgraph_agent_example import graph
with open('assets/langgraph_agent_example.png', 'wb') as f:
    f.write(graph.get_graph().draw_mermaid_png())
"
```
