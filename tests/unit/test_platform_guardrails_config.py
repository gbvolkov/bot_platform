from __future__ import annotations

import json

import pytest

from platform_guardrails.config import (
    GuardrailConfigError,
    inline_guardrail_config_keys,
    resolve_guardrail_policy,
)


def test_resolve_guardrail_policy_maps_policy_to_initialize_kwargs(tmp_path):
    path = tmp_path / "policies.json"
    path.write_text(
        json.dumps(
            {
                "policies": {
                    "sample": {
                        "privacy_enabled": True,
                        "scanners_enabled": False,
                        "tool_execution_enabled": True,
                        "privacy": {
                            "locale": "ru-RU",
                            "entity_replacements": {"RU_PERSON": "typed_placeholder"},
                            "palimpsest_session_options": {"placeholder_style": "typed"},
                        },
                        "scanners": {
                            "failure_policy": "fail_open",
                            "banned_topics": ["topic"],
                            "composite_input_scanners": ["PromptInjection"],
                            "composite_recent_message_limit": 8,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    kwargs = resolve_guardrail_policy("sample", path=path)

    assert kwargs["guardrail_privacy_enabled"] is True
    assert kwargs["guardrail_scanners_enabled"] is False
    assert kwargs["guardrail_tool_execution_enabled"] is True
    assert kwargs["guardrails_locale"] == "ru-RU"
    assert kwargs["guardrail_palimpsest_entity_replacements"] == {
        "RU_PERSON": "typed_placeholder"
    }
    assert kwargs["guardrail_palimpsest_session_options"] == {
        "placeholder_style": "typed"
    }
    assert kwargs["guardrail_scanner_failure_policy"] == "fail_open"
    assert kwargs["guardrail_banned_topics"] == ["topic"]
    assert kwargs["guardrail_composite_input_scanners"] == ("PromptInjection",)
    assert kwargs["guardrail_composite_recent_message_limit"] == 8


def test_resolve_guardrail_policy_rejects_unknown_policy(tmp_path):
    path = tmp_path / "policies.json"
    path.write_text(json.dumps({"policies": {}}), encoding="utf-8")

    with pytest.raises(GuardrailConfigError, match="Unknown guardrail policy"):
        resolve_guardrail_policy("missing", path=path)


def test_inline_guardrail_config_keys_detects_removed_load_json_fields():
    assert inline_guardrail_config_keys(
        {
            "guardrail_policy": "sample",
            "guardrail_privacy_enabled": True,
            "guardrail_tool_profiles": {},
            "provider": "openai",
        }
    ) == {"guardrail_privacy_enabled", "guardrail_tool_profiles"}
