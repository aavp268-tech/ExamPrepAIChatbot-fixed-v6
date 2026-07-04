# src/study_tools/summary_generator.py

from src.study_tools.base_chain import run_prompt

def generate_summary(context: str, length: str = "medium") -> str:
    system_prompt = """
You are an exam preparation assistant.
Create clear and accurate summaries from PDF content.
Use only the given context.
Do not add outside information.
"""

    user_prompt = """
Summarize the following study material.

Length: {length}

Context:
{context}

Output format:
1. Short Overview
2. Key Points
3. Important Terms
4. Exam-Focused Summary
"""

    return run_prompt(system_prompt, user_prompt, context=context, length=length)