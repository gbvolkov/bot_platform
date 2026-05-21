from __future__ import annotations

"""Deprecated synchronous agent loader.

DEPRECATED: This module is a legacy/debugging convenience only and must not be
used in regular application code. Production service code should use
bot_service.agent_registry, which owns async initialization, caching,
checkpointers, tool wiring, guardrails, and streaming metadata.
"""

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable


# STRICT DEPRECATION NOTE:
# This module intentionally exists outside bot_service.agent_registry and should
# not become a second registry implementation. Keep it small, synchronous, and
# limited to ad-hoc/debug builds where directly calling initialize_agent(...) is
# acceptable.

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_CONFIG_PATH = _REPO_ROOT / "data" / "config" / "bot_service" / "load.json"
_SERVICE_ONLY_PARAMS = frozenset({"checkpoint_saver"})


class ProviderRef:
    """Small provider object compatible with agents that read provider.value."""

    def __init__(self, value: str) -> None:
        self.value = value

    @property
    def name(self) -> str:
        return self.value.upper()

    def __str__(self) -> str:
        return self.value


class AgentLoaderError(RuntimeError):
    """Raised when the deprecated synchronous loader cannot build an agent."""


def build_agent_by_id(
    agent_id: str,
    *,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Synchronously import an agent module and call initialize_agent(...).

    The selected config entry's ``params`` are passed to initialize_agent.
    Top-level ``init_context`` is also passed when present. Keys ending in
    ``_path`` are resolved relative to the repository root and loaded as text,
    matching the service registry convention.

    Service-only params such as ``checkpoint_saver`` are ignored unless supplied
    through ``overrides`` with a concrete runtime object.
    """

    entry = get_agent_entry(agent_id, config_path=config_path)
    module_path = _require_str(entry.get("module") or entry.get("path"), "module")
    initialize_agent = _import_initialize_agent(module_path)

    params = _agent_params(entry)
    init_context = _agent_init_context(entry)
    if init_context:
        params["init_context"] = init_context
    explicit_overrides = dict(overrides or {})
    params.update(explicit_overrides)
    params = _coerce_provider(params)
    params = _filter_params_for_callable(initialize_agent, params)
    return initialize_agent(**params)


def get_agent_entry(
    agent_id: str,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a shallow copy of the raw config entry for ``agent_id``."""

    config = _load_config(_resolve_config_path(config_path))
    for entry in _agent_entries(config):
        if entry.get("id") == agent_id:
            return dict(entry)
    available = ", ".join(list_agent_ids(config_path=config_path))
    raise KeyError(f"Unknown agent '{agent_id}'. Available agents: {available}")


def list_agent_ids(*, config_path: str | Path | None = None) -> list[str]:
    """List agent ids from the configured JSON registry file."""

    config = _load_config(_resolve_config_path(config_path))
    ids: list[str] = []
    for entry in _agent_entries(config):
        raw_id = entry.get("id")
        if isinstance(raw_id, str) and raw_id:
            ids.append(raw_id)
    return ids


def load_agent(
    agent_id: str,
    *,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Alias for build_agent_by_id(...)."""

    return build_agent_by_id(
        agent_id,
        config_path=config_path,
        overrides=overrides,
    )


def _resolve_config_path(raw_path: str | Path | None) -> Path:
    if raw_path is None:
        return DEFAULT_AGENT_CONFIG_PATH
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AgentLoaderError(f"Agent config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AgentLoaderError(f"Agent config must be JSON: {path}") from exc
    if not isinstance(data, dict):
        raise AgentLoaderError(f"Agent config root must be an object: {path}")
    return data


def _agent_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = config.get("agents") or config.get("modules")
    if not isinstance(entries, list):
        raise AgentLoaderError("Agent config must contain an 'agents' list.")
    parsed: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict):
            parsed.append(entry)
    return parsed


def _agent_params(entry: dict[str, Any]) -> dict[str, Any]:
    raw_params = entry.get("params") or {}
    if not isinstance(raw_params, dict):
        agent_id = entry.get("id", "<unknown>")
        raise AgentLoaderError(f"Agent '{agent_id}' params must be an object.")

    params = dict(raw_params)
    for name in _SERVICE_ONLY_PARAMS:
        params.pop(name, None)
    return params


def _agent_init_context(entry: dict[str, Any]) -> dict[str, Any]:
    raw_context = entry.get("init_context") or {}
    if not isinstance(raw_context, dict):
        agent_id = entry.get("id", "<unknown>")
        raise AgentLoaderError(f"Agent '{agent_id}' init_context must be an object.")

    resolved: dict[str, Any] = {}
    for raw_key, raw_value in raw_context.items():
        key = _require_str(raw_key, "init_context key").strip()
        if key.endswith("_path"):
            resolved_key = key[:-5]
            if not resolved_key:
                raise AgentLoaderError(f"Agent init_context key '{key}' is invalid.")
            if resolved_key in raw_context or resolved_key in resolved:
                raise AgentLoaderError(
                    "Agent init_context cannot contain both "
                    f"'{resolved_key}' and '{key}'."
                )
            path_value = _require_str(raw_value, f"init_context.{key}")
            path = Path(path_value)
            if not path.is_absolute():
                path = (_REPO_ROOT / path).resolve()
            if not path.is_file():
                raise AgentLoaderError(f"Agent init_context path is not a file: {path}")
            resolved[resolved_key] = path.read_text(encoding="utf-8")
            continue

        if key in resolved:
            raise AgentLoaderError(f"Agent init_context contains duplicate key '{key}'.")
        resolved[key] = raw_value
    return resolved


def _import_initialize_agent(module_path: str) -> Callable[..., Any]:
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise AgentLoaderError(f"Could not import agent module '{module_path}'.") from exc

    initialize_agent = getattr(module, "initialize_agent", None)
    if not callable(initialize_agent):
        raise AgentLoaderError(f"{module_path} does not expose initialize_agent().")
    return initialize_agent


def _coerce_provider(params: dict[str, Any]) -> dict[str, Any]:
    provider = params.get("provider")
    if isinstance(provider, str):
        return {**params, "provider": ProviderRef(provider)}
    return params


def _filter_params_for_callable(
    func: Callable[..., Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(func)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return params
    return {key: value for key, value in params.items() if key in signature.parameters}


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentLoaderError(f"Agent config field '{field_name}' must be a string.")
    return value
