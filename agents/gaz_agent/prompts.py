from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from .locales import get_locale


SYSTEM_PROMPT_VERSION = "v0.6.0"
NODE_PROMPT_SET_VERSION = "v0.6.0"
SUMMARY_PROMPT_VERSION = "v0.6.0"



def get_system_prompt(locale: str) -> str:
    return get_locale(locale)["prompts"]["system"]



def get_prompt(locale: str, key: str) -> str:
    return get_locale(locale)["prompts"][key]



def get_text(locale: str, key: str) -> str:
    return get_locale(locale)["agent"][key]



def render_state_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)



def compose_prompt(
    locale: str,
    key: str,
    payload: Dict[str, Any],
    extra_sections: Optional[Iterable[str]] = None,
) -> str:
    sections = [get_system_prompt(locale), get_prompt(locale, key)]
    sections.extend(section for section in (extra_sections or []) if section)
    sections.extend(["STATE PAYLOAD:", render_state_payload(payload)])
    return "\n\n".join(sections)
