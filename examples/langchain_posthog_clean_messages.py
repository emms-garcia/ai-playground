import datetime
import os
import uuid
from typing import Any, override
from uuid import UUID

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from posthog.ai.langchain import CallbackHandler
from posthog.client import Client as PostHogClient
from pydantic import SecretStr


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


class PostHogCallbackHandler(CallbackHandler):
    """PostHog tracing handler that patches the root trace's Input / Output Messages.

    The base `CallbackHandler` derives the root `$ai_trace` Input / Output from the raw
    first / last chain events. For a tool-calling agent those events are intermediate states —
    the Input can be the full message history and the Output an empty tool-call stub or a
    reasoning-only message — so the PostHog AI Observability dashboard shows tool messages /
    reasoning instead of the user's question and the assistant's final answer.

    This subclass rewrites the root run (`parent_run_id is None`) so the Input is this turn's
    last `HumanMessage` and the Output is this turn's last non-empty `AIMessage`. It captures
    the root once and drops later root chain events so post-agent generations don't overwrite it.
    """

    _root_captured: bool = False

    @staticmethod
    def _last_message(state: object, message_type: type[BaseMessage], role: str) -> dict[str, str] | None:
        """Return the last message of the given type as a chat dict under the given role, or None."""
        messages = state.get("messages") if isinstance(state, dict) else None
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, message_type) and message.text:
                    return {"role": role, "content": message.text}
        return None

    @override
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Scope the root trace's input to this turn's user message; drop chain starts after capture."""
        if self._root_captured:
            return
        if parent_run_id is None:
            user_message = self._last_message(inputs, HumanMessage, "user")
            inputs = [user_message] if user_message else inputs
        super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @override
    def on_chain_end(self, outputs: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        """Scope the root trace's output to this turn's final assistant message."""
        if self._root_captured:
            return
        if parent_run_id is None:
            self._root_captured = True
            ai_message = self._last_message(outputs, AIMessage, "assistant")
            outputs = [ai_message] if ai_message else outputs
        super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)


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
    # A fresh handler per turn: `_root_captured` scopes the root trace to a single turn.
    handler = PostHogCallbackHandler(
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
