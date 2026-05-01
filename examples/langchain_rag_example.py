import os
from typing import cast

from pydantic import SecretStr

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DOCS = [
    Document(
        page_content=(
            "Mark Grayson is the teenage hero Invincible. He is the son of Nolan Grayson, "
            "better known as Omni-Man, and Debbie Grayson. Mark develops Viltrumite powers "
            "later than expected and spends much of the story balancing superhero work, "
            "school, and the pressure of living up to his father's reputation."
        ),
        metadata={"source": "docs/mark_grayson.md"},
    ),
    Document(
        page_content=(
            "Viltrumites are an expansionist warrior species with immense strength, speed, "
            "flight, and durability. Their empire uses infiltration and conquest to absorb "
            "other worlds. Omni-Man was originally sent to Earth as an advance agent for the "
            "Viltrum Empire, which creates the central conflict with Mark."
        ),
        metadata={"source": "docs/viltrumites.md"},
    ),
    Document(
        page_content=(
            "The Guardians of the Globe are Earth's premier superhero team at the start of the "
            "story. Omni-Man secretly murders the original team, an event that exposes the gap "
            "between his public image as Earth's protector and his real loyalty to Viltrum."
        ),
        metadata={"source": "docs/guardians.md"},
    ),
    Document(
        page_content=(
            "Atom Eve is one of Mark's closest allies and eventually one of the most important "
            "relationships in his life. Her powers center on matter manipulation, which makes "
            "her versatile in both combat and reconstruction. Unlike many heroes, she also tries "
            "to use her powers for ordinary people outside pure superhero fights."
        ),
        metadata={"source": "docs/atom_eve.md"},
    ),
    Document(
        page_content=(
            "Allen the Alien works for the Coalition of Planets, a group resisting the Viltrum "
            "Empire. He initially evaluates worlds for their defensive strength but becomes one "
            "of Mark's strongest allies. Allen is important because he helps connect Earth's "
            "problems to the much larger interplanetary war against Viltrum."
        ),
        metadata={"source": "docs/coalition.md"},
    ),
    Document(
        page_content=(
            "Invincible's world mixes grounded family drama with extreme superhero violence. "
            "Major recurring threats include the Flaxans, the Sequids, Angstrom Levy, and the "
            "Viltrum Empire itself. A core theme of the story is whether Mark can remain humane "
            "while inheriting powers and conflicts from a brutal imperial legacy."
        ),
        metadata={"source": "docs/themes_and_threats.md"},
    ),
]

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=160, chunk_overlap=30)
DOCS = SPLITTER.split_documents(RAW_DOCS)
RETRIEVER = BM25Retriever.from_documents(DOCS, k=3)


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents found."

    return "\n\n".join(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs)


@tool(response_format="content_and_artifact")
def search_docs(query: str) -> tuple[str, list[Document]]:
    """Search the Invincible lore docs.

    Use this for questions about Invincible characters, factions, conflicts, themes, or worldbuilding.
    """
    docs = RETRIEVER.invoke(query)
    return format_docs(docs), docs


LLM = init_chat_model(model="llama-3.1-8b-instant", api_key=SecretStr(os.environ["GROQ_API_KEY"]), model_provider="groq")

SYSTEM_PROMPT = """
You are a concise assistant for the Invincible lore docs.

Rules:
- For any question about Invincible lore in this example, call `search_docs` before answering.
- Answer only from the tool results.
- If the tool results do not contain the answer, say you do not know from the docs.
- Cite sources inline using their `source` paths.

Response style:
- Be concise and factual.
- Use the tool results to answer in your own words.
- Do not add any information that is not in the tool results.
"""

agent = create_agent(LLM, [search_docs], system_prompt=SYSTEM_PROMPT)


def run_app(user_input: str) -> list[Document]:
    retrieved_docs: list[Document] = []
    response_parts: list[str] = []

    for item in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="messages",
    ):
        match cast(tuple[BaseMessage, dict], item):
            case (ToolMessage(name="search_docs") as message, _):
                retrieved_docs = cast(list[Document], message.artifact)
                print("\n--- retrieved context ---")
                print(format_docs(retrieved_docs))
                print("-------------------------\n")
            case (BaseMessage() as message, {"langgraph_node": "model"}):
                if isinstance(message.content, str):
                    response_parts.append(message.content)

    print("".join(response_parts))
    return retrieved_docs


if __name__ == "__main__":
    print("Ask questions about the Invincible lore docs.")
    print("Try: who is Mark Grayson?")
    print()

    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            break

        print(run_app(user_input))
