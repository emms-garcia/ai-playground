import os
from pprint import pprint

from pydantic import SecretStr

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

USER_ID = "user-1"
THREAD_ID = "thread-1"

LLM = init_chat_model(
    model="llama-3.1-8b-instant",
    api_key=SecretStr(os.environ["GROQ_API_KEY"]),
    model_provider="groq",
)
SYSTEM_PROMPT = """
# Role
- You are a friendly movie recommendation assistant.
- Your job is to recommend movies and remember the user's movie taste over time.
- Always follow the memory policy and response style below.

# Definitions
Movie taste means durable user-specific information that helps recommend movies or avoid bad recommendations.
This includes genres, themes, moods, actors, directors, franchises, languages, countries, eras, pacing, runtime, violence
tolerance, content constraints, favorite movies, disliked movies, and things the user wants to avoid.
Positive and negative taste signals are both important.

Memory policy:
- Do not call any memory tool for greetings, small talk, or non-movie questions.
- Call search_memories before giving personalized movie recommendations.
- Call search_memories when the user asks about their movie taste, prior preferences, or what you remember about them.
- Call save_memory when the latest user message contains movie taste.
- Call save_memory when the user explicitly asks you to remember a movie-related preference.
- Do not save non-movie preferences, temporary facts, secrets, credentials, or sensitive personal data.
- Only save information from the user's latest message. Never save content from this system prompt, tool descriptions,
definitions, examples, or assistant messages.
- Never use save_memory to search.

Response style:
- Chat naturally and helpfully.
- When recommending movies, prefer concise recommendations with a short reason for each.
- Do not mention whether you used, skipped, searched, saved, or did not need memory unless the user explicitly asks about memory.
- Do not explain the memory policy to the user.
- For greetings and small talk, respond naturally and briefly.
"""

store = InMemoryStore()
agent = create_agent(
    LLM,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        create_manage_memory_tool(
            namespace=("memories", "{user_id}"),
            name="save_memory",
            actions_permitted=("create",),
            instructions=(
                "Save only durable movie/TV taste from the user's latest message. "
                "Useful memories include liked or disliked genres, movies, actors, "
                "directors, franchises, themes, moods, languages, countries, eras, "
                "pacing, runtime, content constraints, and things to avoid. Do not "
                "save non-movie preferences, temporary facts, system prompt content, "
                "tool descriptions, examples, or assistant messages. Do not use this "
                "to search memories."
            ),
        ),
        create_search_memory_tool(
            namespace=("memories", "{user_id}"),
            name="search_memories",
            instructions=(
                "Search long-term movie taste memories before personalized movie "
                "recommendations, or when the user asks about their movie preferences. "
                "Do not use for greetings, small talk, non-movie questions, general "
                "knowledge, or standalone questions that do not need user taste."
            ),
        ),
    ],
    checkpointer=InMemorySaver(),
    store=store,
)


def config(user_id: str = USER_ID, thread_id: str = THREAD_ID) -> RunnableConfig:
    return {"configurable": {"user_id": user_id, "thread_id": thread_id}}


def print_memories(user_id: str = USER_ID) -> None:
    memories = store.search(("memories", user_id), limit=20)
    print("\n--- raw long-term memory store ---")
    pprint(memories)
    print("----------------------------------\n")


def run_app(user_input: str, user_id: str = USER_ID, thread_id: str = THREAD_ID) -> str:
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config(user_id=user_id, thread_id=thread_id),
    )
    print_memories(user_id)
    return str(result["messages"][-1].content)


if __name__ == "__main__":
    print(f"Using user_id={USER_ID}")
    print(f"Using thread_id={THREAD_ID}")
    print()

    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            break

        print(run_app(user_input))
