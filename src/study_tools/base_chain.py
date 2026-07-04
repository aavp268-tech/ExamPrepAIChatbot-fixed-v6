# src/study_tools/base_chain.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

def run_prompt(system_prompt: str, user_prompt: str, **kwargs) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke(kwargs)