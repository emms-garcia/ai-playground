import os
import uuid
import datetime
from pydantic import SecretStr

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


@tool
def get_current_time() -> str:
    """Returns the current UTC time."""
    time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"The current time is: {time}"


@tool
def multiply(a: float, b: float) -> str:
    """Multiplies two numbers together."""
    result = a * b
    return f"The result of {a} * {b} is: {result}"


SYSTEM_PROMPT = """
You are a concise assistant. Use tools to help answer the user's question when appropriate. Here are the tools you can use:
- When the user asks for the curren time, call the `get_current_time` tool.
- When the user asks to multiply two numbers, call the `multiply` tool with the appropriate arguments.

Use the tool responds to build your response.
"""


agent = create_agent(
    init_chat_model(
        model="openai/gpt-oss-20b",
        model_provider="groq",
        api_key=SecretStr(os.environ["GROQ_API_KEY"]),
    ),
    [get_current_time, multiply],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)


def run(user_input: str, session_id: str) -> None:
    print("Assistant: ", end="", flush=True)
    for part in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode=["messages"],
        config={"configurable": {"thread_id": session_id}},
        version="v2",
    ):
        if part["type"] == "messages":
            message, _ = part["data"]
            if isinstance(message, AIMessage) and not message.tool_calls:
                print(message.content, end="", flush=True)
    print()


if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        run(user_input, session_id)
