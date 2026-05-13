from __future__ import annotations

import uuid
from typing import Any, Literal, Mapping, NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config


RiskLevel = Literal["low", "medium", "high", "critical"]


class GuardrailContext(TypedDict):
    tenant_id: str | None
    user_id: str | None
    user_role: str
    thread_id: str
    agent_name: str
    route: str | None
    model: str | None
    tool_name: str | None
    request_id: str
    risk_level: RiskLevel
    allow_deanonymization: bool
    allow_external_tool_access: bool
    allow_external_search: NotRequired[bool]
    allow_file_export: bool
    allow_sensitive_data: bool


_MISSING_TENANT = "__no_tenant__"
_MISSING_USER = "__no_user__"
_MISSING_THREAD = "__manual__"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _configurable_from_config(config: RunnableConfig | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    return _as_mapping(config.get("configurable"))


def _config_from_runtime(runtime: Any) -> Mapping[str, Any]:
    config = getattr(runtime, "config", None)
    if isinstance(config, Mapping):
        return config
    try:
        current = get_config()
    except Exception:
        current = None
    return current if isinstance(current, Mapping) else {}


def _first_text(*values: Any, default: str | None = None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _risk_level(value: Any) -> RiskLevel:
    if value in {"low", "medium", "high", "critical"}:
        return value
    return "low"


def build_guardrail_context(
    *,
    config: RunnableConfig | Mapping[str, Any] | None = None,
    runtime: Any = None,
    state: Mapping[str, Any] | None = None,
    agent_name: str | None = None,
    route: str | None = None,
    model: str | None = None,
    tool_name: str | None = None,
    request_id: str | None = None,
    default_allow_deanonymization: bool = True,
) -> GuardrailContext:
    config_mapping = config if isinstance(config, Mapping) else _config_from_runtime(runtime)
    configurable = _configurable_from_config(config_mapping)
    state_mapping = _as_mapping(state)

    execution_info = getattr(runtime, "execution_info", None)
    runtime_thread_id = getattr(execution_info, "thread_id", None)

    tenant_id = _first_text(
        configurable.get("tenant_id"),
        state_mapping.get("tenant_id"),
    )
    user_id = _first_text(
        configurable.get("user_id"),
        state_mapping.get("user_id"),
    )
    thread_id = _first_text(
        configurable.get("thread_id"),
        runtime_thread_id,
        state_mapping.get("thread_id"),
        default=_MISSING_THREAD,
    )

    allow_external_tool_access = _bool_value(
        configurable.get("allow_external_tool_access", configurable.get("allow_external_search")),
        False,
    )

    return {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "user_role": _first_text(
            configurable.get("user_role"),
            state_mapping.get("user_role"),
            default="default",
        )
        or "default",
        "thread_id": thread_id or _MISSING_THREAD,
        "agent_name": agent_name or _first_text(state_mapping.get("agent_name"), default="unknown") or "unknown",
        "route": route or _first_text(state_mapping.get("route")),
        "model": model or _first_text(configurable.get("model"), state_mapping.get("model")),
        "tool_name": tool_name,
        "request_id": request_id
        or _first_text(configurable.get("request_id"), state_mapping.get("request_id"))
        or f"gr-{uuid.uuid4().hex}",
        "risk_level": _risk_level(configurable.get("risk_level") or state_mapping.get("risk_level")),
        "allow_deanonymization": _bool_value(
            configurable.get("allow_deanonymization"),
            default_allow_deanonymization,
        ),
        "allow_external_tool_access": allow_external_tool_access,
        "allow_external_search": allow_external_tool_access,
        "allow_file_export": _bool_value(configurable.get("allow_file_export"), False),
        "allow_sensitive_data": _bool_value(configurable.get("allow_sensitive_data"), False),
    }


def _scope_part(value: str | None, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return text.replace("|", "_")


def privacy_scope_key(context: GuardrailContext | Mapping[str, Any]) -> str:
    return "|".join(
        [
            _scope_part(context.get("tenant_id"), _MISSING_TENANT),
            _scope_part(context.get("user_id"), _MISSING_USER),
            _scope_part(context.get("thread_id"), _MISSING_THREAD),
        ]
    )


__all__ = [
    "GuardrailContext",
    "RiskLevel",
    "build_guardrail_context",
    "privacy_scope_key",
]
