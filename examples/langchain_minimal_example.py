import os

from pydantic import SecretStr

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = "You are a concise assistant. Chat with the user naturally."

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", "{input}")])

LLM = init_chat_model(
    model="llama-3.1-8b-instant",
    api_key=SecretStr(os.environ["GROQ_API_KEY"]),
    model_provider="groq",
)

chain = prompt | LLM | StrOutputParser()

if __name__ == "__main__":
    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            break

        print(chain.invoke({"input": user_input}))
