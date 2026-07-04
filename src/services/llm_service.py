from src.model.llm import get_llm
from src.utils.logger import get_logger

logger = get_logger()
llm = get_llm()

def generate_response(prompt, temperature=0.7, max_tokens=None):
    try:
        if max_tokens is not None:
            response = get_llm(temperature=temperature, max_tokens=max_tokens).invoke(prompt)
        else:
            response = llm.invoke(prompt)

        logger.info(f"LLM response generated — {len(response.content)} chars")
        return response.content

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise RuntimeError(f"LLM generation failed: {str(e)}")