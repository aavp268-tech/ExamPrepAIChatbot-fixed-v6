# src/chatbot/chatbot.py

from src.study_tools.base_chain import run_prompt

def generate_chatbot_answer(question: str, context: str, chat_history: str = "") -> str:
    system_prompt = """
You are an exam preparation chatbot using RAG.
Answer only from the retrieved PDF context.
If the answer is not present in the context, say:
"I could not find this answer in the uploaded PDF."
Keep the answer clear, helpful, and exam focused.
"""

    user_prompt = """
Chat History:
{chat_history}

Retrieved PDF Context:
{context}

Student Question:
{question}

Answer:
"""

    return run_prompt(
        system_prompt,
        user_prompt,
        question=question,
        context=context,
        chat_history=chat_history,
    )