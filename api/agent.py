import datetime
import os
from typing import Annotated

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from pydantic import SecretStr
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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


_PROMPT = """\
# Role
AI Assistant.

# Task
Use the best tool(s) to answer the user's question.If the user's question is ambiguous, ask for clarification.
List the tools you have access to and their capabilities in your initial response.
"""
_MODEL = init_chat_model(model="llama-3.1-8b-instant", api_key=SecretStr(os.environ["GROQ_API_KEY"]), model_provider="groq")
AGENT = create_agent(
    _MODEL,
    [get_current_time, multiply],
    system_prompt=_PROMPT,
    checkpointer=InMemorySaver(),
)
