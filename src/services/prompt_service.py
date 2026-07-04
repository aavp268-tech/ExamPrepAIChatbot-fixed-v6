# FIXED — cache loaded prompts + readable error on missing file
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=32)
def load_prompt(path):
    prompt_path = BASE_DIR / path
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Make sure Member 7 has created this file."
        )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
