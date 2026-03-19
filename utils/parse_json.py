import json
import re


def parse_json(text: str) -> dict | list:
    """
    Parse JSON from LLM response with error recovery.

    Args:
        text: Raw LLM response

    Returns:
        Parsed JSON (dict or list)
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks (dict or array)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object or array
    json_match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"⚠️ Could not parse JSON from response: {text[:200]}...")
    return {}
