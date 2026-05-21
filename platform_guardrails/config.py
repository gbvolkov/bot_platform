from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .url_policy import coerce_url_policy_config


DEFAULT_GUARDRAIL_POLICY_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "config"
    / "guardrails"
    / "policies.json"
)

INLINE_GUARDRAIL_CONFIG_KEYS = frozenset(
    {
        "guardrails_locale",
        "guardrail_privacy_enabled",
        "guardrail_scanners_enabled",
        "guardrail_tool_execution_enabled",
        "guardrail_scanner_failure_policy",
        "guardrail_banned_topics",
        "guardrail_prompt_injection_model",
        "guardrail_prompt_injection_model_revision",
        "guardrail_prompt_injection_threshold",
        "guardrail_url_policy",
        "guardrail_scan_system_prompt",
        "guardrail_verbose_logging",
        "guardrail_composite_input_scanners",
        "guardrail_composite_recent_message_limit",
        "guardrail_palimpsest_run_entities",
        "guardrail_palimpsest_entity_replacements",
        "guardrail_palimpsest_options",
        "guardrail_palimpsest_session_options",
        "guardrail_tool_profiles",
        "guardrail_unprofiled_tools",
    }
)


class GuardrailConfigError(ValueError):
    """Raised when guardrail policy configuration is invalid."""


def load_guardrail_policy_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = DEFAULT_GUARDRAIL_POLICY_CONFIG_PATH if path is None else Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GuardrailConfigError("Guardrail policy config must be a JSON object.")
    policies = data.get("policies")
    if not isinstance(policies, dict):
        raise GuardrailConfigError("Guardrail policy config must contain a 'policies' object.")
    return data


def resolve_guardrail_policy(
    policy_id: str,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise GuardrailConfigError("guardrail_policy must be a non-empty string.")
    config = load_guardrail_policy_config(path)
    policies = config["policies"]
    policy = policies.get(policy_id)
    if not isinstance(policy, dict):
        raise GuardrailConfigError(f"Unknown guardrail policy: {policy_id!r}.")
    return guardrail_policy_to_init_kwargs(policy, policy_id=policy_id)


def guardrail_policy_to_init_kwargs(policy: Mapping[str, Any], *, policy_id: str = "<inline>") -> dict[str, Any]:
    privacy_enabled = _require_bool(policy.get("privacy_enabled", False), f"{policy_id}.privacy_enabled")
    scanners_enabled = _require_bool(policy.get("scanners_enabled", False), f"{policy_id}.scanners_enabled")
    tool_execution_enabled = _require_bool(
        policy.get("tool_execution_enabled", False),
        f"{policy_id}.tool_execution_enabled",
    )

    kwargs: dict[str, Any] = {
        "guardrail_privacy_enabled": privacy_enabled,
        "guardrail_scanners_enabled": scanners_enabled,
        "guardrail_tool_execution_enabled": tool_execution_enabled,
    }

    privacy = policy.get("privacy") or {}
    if not isinstance(privacy, Mapping):
        raise GuardrailConfigError(f"{policy_id}.privacy must be an object.")
    if "locale" in privacy:
        kwargs["guardrails_locale"] = _require_str(privacy["locale"], f"{policy_id}.privacy.locale")
    if privacy.get("run_entities") is not None:
        kwargs["guardrail_palimpsest_run_entities"] = _require_list(
            privacy["run_entities"],
            f"{policy_id}.privacy.run_entities",
        )
    if privacy.get("entity_replacements") is not None:
        replacements = privacy["entity_replacements"]
        if not isinstance(replacements, (dict, list)):
            raise GuardrailConfigError(f"{policy_id}.privacy.entity_replacements must be an object or list.")
        kwargs["guardrail_palimpsest_entity_replacements"] = replacements
    if privacy.get("palimpsest_options") is not None:
        kwargs["guardrail_palimpsest_options"] = _require_dict(
            privacy["palimpsest_options"],
            f"{policy_id}.privacy.palimpsest_options",
        )
    if privacy.get("palimpsest_session_options") is not None:
        kwargs["guardrail_palimpsest_session_options"] = _require_dict(
            privacy["palimpsest_session_options"],
            f"{policy_id}.privacy.palimpsest_session_options",
        )

    scanners = policy.get("scanners") or {}
    if not isinstance(scanners, Mapping):
        raise GuardrailConfigError(f"{policy_id}.scanners must be an object.")
    logging_config = policy.get("logging") or {}
    if not isinstance(logging_config, Mapping):
        raise GuardrailConfigError(f"{policy_id}.logging must be an object.")
    if logging_config.get("verbose") is not None:
        kwargs["guardrail_verbose_logging"] = _require_bool(
            logging_config["verbose"],
            f"{policy_id}.logging.verbose",
        )
    if scanners.get("failure_policy") is not None:
        kwargs["guardrail_scanner_failure_policy"] = _require_str(
            scanners["failure_policy"],
            f"{policy_id}.scanners.failure_policy",
        )
    if scanners.get("banned_topics") is not None:
        kwargs["guardrail_banned_topics"] = _require_list(
            scanners["banned_topics"],
            f"{policy_id}.scanners.banned_topics",
        )
    if scanners.get("prompt_injection_model") is not None:
        model = scanners["prompt_injection_model"]
        if not isinstance(model, (str, dict)):
            raise GuardrailConfigError(f"{policy_id}.scanners.prompt_injection_model must be a string or object.")
        kwargs["guardrail_prompt_injection_model"] = model
    if scanners.get("prompt_injection_model_revision") is not None:
        kwargs["guardrail_prompt_injection_model_revision"] = _require_str(
            scanners["prompt_injection_model_revision"],
            f"{policy_id}.scanners.prompt_injection_model_revision",
        )
    if scanners.get("prompt_injection_threshold") is not None:
        kwargs["guardrail_prompt_injection_threshold"] = float(scanners["prompt_injection_threshold"])
    if scanners.get("url_policy") is not None:
        try:
            url_policy = coerce_url_policy_config(
                _require_dict(scanners["url_policy"], f"{policy_id}.scanners.url_policy")
            )
        except (TypeError, ValueError) as exc:
            raise GuardrailConfigError(
                f"Invalid guardrail policy field '{policy_id}.scanners.url_policy': {exc}"
            ) from exc
        if url_policy is not None:
            kwargs["guardrail_url_policy"] = url_policy
    if scanners.get("scan_system_prompt") is not None:
        kwargs["guardrail_scan_system_prompt"] = _require_bool(
            scanners["scan_system_prompt"],
            f"{policy_id}.scanners.scan_system_prompt",
        )
    if scanners.get("composite_input_scanners") is not None:
        kwargs["guardrail_composite_input_scanners"] = tuple(
            _require_list(
                scanners["composite_input_scanners"],
                f"{policy_id}.scanners.composite_input_scanners",
            )
        )
    if scanners.get("composite_recent_message_limit") is not None:
        kwargs["guardrail_composite_recent_message_limit"] = int(scanners["composite_recent_message_limit"])
    return kwargs


def inline_guardrail_config_keys(params: Mapping[str, Any]) -> set[str]:
    return set(params).intersection(INLINE_GUARDRAIL_CONFIG_KEYS)


def _require_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise GuardrailConfigError(f"Guardrail policy field '{field_name}' must be a boolean.")


def _require_str(value: Any, field_name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise GuardrailConfigError(f"Guardrail policy field '{field_name}' must be a non-empty string.")


def _require_list(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    raise GuardrailConfigError(f"Guardrail policy field '{field_name}' must be a list.")


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raise GuardrailConfigError(f"Guardrail policy field '{field_name}' must be an object.")


__all__ = [
    "DEFAULT_GUARDRAIL_POLICY_CONFIG_PATH",
    "GuardrailConfigError",
    "INLINE_GUARDRAIL_CONFIG_KEYS",
    "guardrail_policy_to_init_kwargs",
    "inline_guardrail_config_keys",
    "load_guardrail_policy_config",
    "resolve_guardrail_policy",
]
