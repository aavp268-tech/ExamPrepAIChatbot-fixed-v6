# src/study_tools/revision_sheet_generator.py

from src.study_tools.base_chain import run_prompt

def generate_revision_sheet(context: str) -> str:
    system_prompt = """
You create last-minute revision sheets for exams.
The output must be concise, high-yield, and easy to revise.
Use only the provided context.
"""

    user_prompt = """
Create a revision sheet from this content:

{context}

Output format:
# Quick Revision Sheet
## Must-Know Points
## Important Definitions
## Formulas / Rules
## Common Mistakes
## Final Recap
"""

    return run_prompt(system_prompt, user_prompt, context=context)