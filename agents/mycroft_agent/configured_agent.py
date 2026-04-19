from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from deepagents.backends import FilesystemBackend

from agents.utils import ModelType

from .agent import initialize_agent as initialize_mycroft_agent
from .cli_config import (
    build_internal_tools,
    load_cli_config,
    load_mcp_tools_from_config,
    validate_required_environment,
)
from .subagent_loader import initialize_configured_subagents


_REPO_ROOT = Path(__file__).resolve().parents[2]


class PosixVirtualFilesystemBackend(FilesystemBackend):
    def ls_info(self, path: str) -> list[dict[str, Any]]:
        items = super().ls_info(path)
        if not self.virtual_mode:
            return items

        normalized: list[dict[str, Any]] = []
        for item in items:
            path_value = item.get("path")
            if not isinstance(path_value, str):
                normalized.append(item)
                continue
            normalized_path = path_value.replace("\\", "/")
            while normalized_path.startswith("//"):
                normalized_path = normalized_path[1:]
            normalized.append({**item, "path": normalized_path})
        return normalized


def _resolve_config_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def normalize_skill_source(raw_path: str) -> str:
    path = raw_path.replace("\\", "/").strip()
    if not path.startswith("/"):
        path = "/" + path
    return path.rstrip("/") or "/"


def build_skills_backend(skills: tuple[str, ...]) -> FilesystemBackend | None:
    if not skills:
        return None
    return PosixVirtualFilesystemBackend(root_dir=_REPO_ROOT, virtual_mode=True)


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    *,
    config_path: str | Path,
    model_size: str = "base",
    temperature: float = 0.2,
    checkpoint_saver: Any | None = None,
    streaming: bool = False,
    reasoning: str | None = None,
    max_tool_calls: int | None = 12,
) -> Any:
    resolved_config_path = _resolve_config_path(config_path)
    mycroft_config = load_cli_config(resolved_config_path)
    validate_required_environment(mycroft_config, provider.value)

    stateless_subagents = asyncio.run(
        initialize_configured_subagents(mycroft_config.subagents.stateless)
    )
    stateful_subagents = asyncio.run(
        initialize_configured_subagents(mycroft_config.subagents.stateful)
    )
    internal_tools = build_internal_tools(mycroft_config.internal_tools)
    mcp_tools = asyncio.run(load_mcp_tools_from_config(mycroft_config.mcp))

    skills = tuple(normalize_skill_source(skill) for skill in mycroft_config.skills.paths)
    return initialize_mycroft_agent(
        provider=provider,
        model_size=model_size,
        temperature=temperature,
        system_prompt=mycroft_config.system_prompt,
        tools=[*internal_tools, *mcp_tools],
        stateless_subagents=stateless_subagents,
        stateful_subagents=stateful_subagents,
        checkpoint_saver=checkpoint_saver,
        streaming=streaming,
        reasoning=reasoning,
        max_tool_calls=max_tool_calls,
        interrupt_on=mycroft_config.deepagents.interrupt_on or None,
        skills=skills,
        backend=build_skills_backend(skills),
    )
