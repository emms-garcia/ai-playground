import os
import datetime
from typing import cast

from pydantic import SecretStr

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langgraph.store.memory import InMemoryStore


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


# 2. Define model
API_KEY = SecretStr(os.environ["GROQ_API_KEY"])
LLM = init_chat_model(model="llama-3.1-8b-instant", api_key=API_KEY, model_provider="groq")

# 3. Build agent (handles tool call loop automatically)
SYSTEM_PROMPT = """
    Role: AI Assistant.

    Task: Use the best tool(s) to answer the user's question. If the user's question is ambiguous, ask for clarification.
"""
agent = create_agent(LLM, TOOLS, system_prompt=SYSTEM_PROMPT, store=InMemoryStore())


# 4. App entry point
def run_app(user_input: str):
    tool_results = {}
    response_parts = []
    for item in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="messages",
    ):
        match cast(tuple[BaseMessage, dict], item):
            case (ToolMessage(name=str() as name) as message, _):
                tool_results[name] = message.artifact
            case (BaseMessage() as message, {"langgraph_node": "model"}):
                if isinstance(message.content, str):
                    response_parts.append(message.content)

    print("".join(response_parts))
    return tool_results


if __name__ == "__main__":
    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            break

        print(run_app(user_input))
