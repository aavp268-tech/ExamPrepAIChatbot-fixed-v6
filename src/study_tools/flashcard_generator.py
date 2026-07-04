# src/study_tools/flashcard_generator.py

from src.study_tools.base_chain import run_prompt

def generate_flashcards(context: str, count: int = 10) -> str:
    system_prompt = """
You generate useful exam flashcards.
Each flashcard must have a front question and a back answer.
Keep answers short and easy to revise.
Use only the provided context.
"""

    user_prompt = """
Create {count} flashcards from this content:

{context}

Output format:
1. Front: question
   Back: answer
"""

    return run_prompt(system_prompt, user_prompt, context=context, count=count)