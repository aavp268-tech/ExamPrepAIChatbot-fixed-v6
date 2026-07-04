from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.config.setting import LLM_MODEL, TEMPERATURE

load_dotenv()


def get_llm(temperature=TEMPERATURE, max_tokens=512):
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return llm
