# src/study_tools/mcq_generator.py

from src.study_tools.base_chain import run_prompt

def generate_mcqs(context: str, count: int = 10) -> str:
    system_prompt = """
You generate exam-style multiple choice questions.
Each question must have 4 options, one correct answer, and a short explanation.
Use only the provided context.
"""

    user_prompt = """
Generate {count} MCQs from this content:

{context}

Output format:
Q1. Question
A. Option A
B. Option B
C. Option C
D. Option D
Correct Answer: A/B/C/D
Explanation: short explanation
"""

    return run_prompt(system_prompt, user_prompt, context=context, count=count)