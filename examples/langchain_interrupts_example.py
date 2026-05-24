import os
import uuid
from typing import Any, cast

from pydantic import SecretStr

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import GraphOutput
from langgraph.types import Command


@tool
def lookup_account_balance(account_name: str) -> str:
    """Look up a customer's account balance.

    Use this when the user asks about a balance for a specific account.
    Valid demo account names are: checking, savings, and travel.
    """
    fake_balances = {
        "checking": "$1,284.17",
        "savings": "$9,991.02",
        "travel": "$412.55",
    }
    balance = fake_balances.get(account_name.lower())
    if balance is None:
        return f"No account named '{account_name}' was found."
    return f"The balance for the {account_name} account is {balance}."


SYSTEM_PROMPT = """
You are a concise banking support assistant.

Rules:
- If the user asks for an account balance, call `lookup_account_balance`.
- The demo only supports these account names: checking, savings, and travel.
- If the user did not provide an account name, ask a normal follow-up question.
- After a tool returns, answer with the tool result before suggesting anything else.
- Keep responses short and practical.
"""


agent = create_agent(
    init_chat_model(
        model="llama-3.1-8b-instant",
        api_key=SecretStr(os.environ["GROQ_API_KEY"]),
        model_provider="groq",
    ),
    [lookup_account_balance],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "lookup_account_balance": InterruptOnConfig(
                    allowed_decisions=["approve", "edit", "reject"],
                    description="Review account lookup before the tool runs.",
                )
            }
        )
    ],
    checkpointer=InMemorySaver(),
)


def prompt_for_decision(action_request: dict[str, Any], review_config: dict[str, Any]) -> dict[str, Any]:
    tool_name = action_request["name"]
    arguments = action_request["args"]
    allowed_decisions = review_config["allowed_decisions"]

    print("\n\nInterrupt triggered before tool execution.")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {arguments}")
    print(f"Allowed decisions: {', '.join(allowed_decisions)}")
    while True:
        decision = input("Decision [approve/edit/reject]: ").strip().lower()
        if decision == "approve" and "approve" in allowed_decisions:
            return {"type": "approve"}
        if decision == "edit" and "edit" in allowed_decisions:
            updated_arguments = dict(arguments)
            for key, value in arguments.items():
                replacement = input(f"New value for {key} [{value}]: ").strip()
                if replacement:
                    updated_arguments[key] = replacement
            return {"type": "edit", "edited_action": {"name": tool_name, "args": updated_arguments}}
        if decision == "reject" and "reject" in allowed_decisions:
            reason = input("Rejection reason: ").strip()
            return {"type": "reject", "message": reason or "The human reviewer rejected this tool call."}

        print("Invalid decision. Try again.")


def print_assistant_message(result: GraphOutput) -> None:
    messages = result.value["messages"]
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    if tool_messages and isinstance(tool_messages[-1].content, str):
        print(f"Tool result: {tool_messages[-1].content}")
    if messages and isinstance(messages[-1], AIMessage) and isinstance(messages[-1].content, str):
        print(f"Assistant: {messages[-1].content}")


def resume_from_interrupts(interrupts: tuple[Any, ...], config: RunnableConfig) -> None:
    decisions: list[dict[str, Any]] = []
    for interrupt in interrupts:
        payload = interrupt.value
        for action_request, review_config in zip(payload["action_requests"], payload["review_configs"], strict=True):
            decisions.append(prompt_for_decision(action_request, review_config))

    resumed = cast(
        GraphOutput,
        agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            version="v2",
        ),
    )
    print_assistant_message(resumed)


def run_app(user_input: str, thread_id: str) -> None:
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})

    result = cast(
        GraphOutput,
        agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            version="v2",
        ),
    )

    if result.interrupts:
        resume_from_interrupts(result.interrupts, config)
        return

    print_assistant_message(result)


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    print(f"Session ID: {thread_id}")
    print("Try prompts like: 'what is the balance of my checking account?'")
    print("When the tool is about to run, the app will pause and ask for approval.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        run_app(user_input, thread_id)
