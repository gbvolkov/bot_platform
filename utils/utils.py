import json


def is_valid_json_string(s: str) -> bool:
    if not isinstance(s, str):
        return False
    try:
        #s = clean_json_text(s)
        json.loads(s)
        return True
    except (json.JSONDecodeError, TypeError, ValueError):
        return False