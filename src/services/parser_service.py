import json
import re

from src.utils.logger import get_logger

logger = get_logger()


def parse_json_response(content):
    try:
        content = content.strip()
        # strip ```json or plain ```
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"JSON parse failed: {e} | raw: {content[:200]}")
        return {
            "success": False,
            "error": str(e),
            "raw_response": content
        }
