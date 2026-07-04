# src/study_tools/notes_generator.py

from src.study_tools.base_chain import run_prompt

def generate_notes(context: str, topic: str = "the given content") -> str:
    system_prompt = """
You are an expert exam preparation assistant.
Generate structured notes for students.
Use headings, bullet points, definitions, formulas, and examples where useful.
Use ONLY the provided context.
"""

    user_prompt = """
Topic: {topic}

PDF Context:
{context}

Generate exam-ready notes.

Output format:
# Topic Name
## Key Concepts
## Important Definitions
## Explanation
## Examples
## Exam Tips
"""

    return run_prompt(system_prompt, user_prompt, context=context, topic=topic)