import os
import datetime
from typing import Annotated, cast, Literal
from typing_extensions import TypedDict

from pydantic import SecretStr

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore


class AppState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: dict[str, object]


# 1. Define tools
@tool(response_format="content_and_artifact")
def get_current_time() -> tuple[str, str]:
    """Returns the current UTC time.

    Only use this if the user asks for the current time.
    """
    time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"The current time is: {time}", time


@tool(response_format="content_and_artifact")
def multiply(a: float, b: float) -> tuple[str, float]:
    """Multiplies two numbers together.

    Only use this if the user asks you to multiply two numbers.
    """
    result = a * b
    return f"The result of {a} * {b} is: {result}", result


TOOLS = [get_current_time, multiply]

SYSTEM_PROMPT = """
    Role: AI Assistant.

    Task: Use the best tool(s) to answer the user's question. If the user's question is ambiguous, ask for clarification.
"""

# 2. Define model
API_KEY = SecretStr(os.environ["GROQ_API_KEY"])
LLM = init_chat_model(model="llama-3.1-8b-instant", api_key=API_KEY, model_provider="groq").bind_tools(TOOLS)


# 3. Define graph nodes
def model_node(state: AppState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [LLM.invoke(messages)]}


def should_continue(state: AppState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return cast(Literal["__end__"], END)


# 4. Define graph nodes (continued)
def collect_artifacts_node(state: AppState) -> dict:
    new_results = {msg.name: msg.artifact for msg in state["messages"] if isinstance(msg, ToolMessage) and msg.name is not None}
    return {"tool_results": {**state["tool_results"], **new_results}}


# 5. Build graph
tool_node = ToolNode(TOOLS)

graph = (
    StateGraph(AppState)
    .add_node("model", model_node)
    .add_node("tools", tool_node)
    .add_node("collect_artifacts", collect_artifacts_node)
    .set_entry_point("model")
    .add_conditional_edges("model", should_continue)
    .add_edge("tools", "collect_artifacts")
    .add_edge("collect_artifacts", "model")
    .compile(store=InMemoryStore())
)


# 6. App entry point
def run_app(user_input: str) -> dict[str, object]:
    response_parts: list[str] = []
    final_state: AppState | None = None

    initial_state: AppState = {
        "messages": [HumanMessage(content=user_input)],
        "tool_results": {},
    }
    for event_type, data in graph.stream(initial_state, stream_mode=["messages", "values"]):
        match event_type, data:
            case "messages", (BaseMessage() as message, {"langgraph_node": "model"}):
                if isinstance(message.content, str):
                    response_parts.append(message.content)
            case "values", _:
                final_state = cast(AppState, data)

    print("".join(response_parts))
    return final_state["tool_results"] if final_state else {}


if __name__ == "__main__":
    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            break

        print(run_app(user_input))
