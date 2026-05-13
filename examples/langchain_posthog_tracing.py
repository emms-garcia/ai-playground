import os
import uuid
import datetime
from posthog.client import Client as PostHogClient
from posthog.ai.langchain import CallbackHandler
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


def run(user_input: str, ph_client: PostHogClient, session_id: str) -> None:
    handler = CallbackHandler(
        client=ph_client,
        distinct_id="demo-user",
        trace_id=uuid.uuid4(),
        properties={"$ai_session_id": session_id},
    )
    print("Assistant: ", end="", flush=True)
    for part in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode=["messages"],
        config={"callbacks": [handler], "configurable": {"thread_id": session_id}},
        version="v2",
    ):
        if part["type"] == "messages":
            message, _ = part["data"]
            if isinstance(message, AIMessage) and not message.tool_calls:
                print(message.content, end="", flush=True)
    print()


if __name__ == "__main__":
    ph_client = PostHogClient(
        project_api_key=os.environ["POSTHOG_API_KEY"],
        host=os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com"),
    )

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in {"exit", "quit"}:
                break
            run(user_input, ph_client, session_id)
    finally:
        ph_client.shutdown()
