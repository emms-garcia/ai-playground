import os
from pprint import pprint
from langchain.agents import AgentState

from pydantic import SecretStr

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


SYSTEM_PROMPT = """
    Role: AI Assistant.

    Task: Chat naturally with the user. Use the conversation history when answering.
"""

LLM = init_chat_model(
    model="llama-3.1-8b-instant",
    api_key=SecretStr(os.environ["GROQ_API_KEY"]),
    model_provider="groq",
)


def model_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [LLM.invoke(messages)]}


graph = (
    StateGraph(AgentState)
    .add_node("model", model_node)
    .add_edge(START, "model")
    .add_edge("model", END)
    .compile(checkpointer=InMemorySaver())
)


def run_app(user_input: str, thread_id: str = "demo-thread") -> str:
    state = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print("\n---State---")
    pprint(state)
    print("------------------------------\n")
    return str(state["messages"][-1].content)


if __name__ == "__main__":
    thread_id = "demo-thread"
    print(f"Using thread_id={thread_id}")
    print(
        "The in-memory checkpointer preserves this chat while the process is running."
    )
    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            break

        print(run_app(user_input, thread_id=thread_id))
