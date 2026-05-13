from __future__ import annotations

"""Compatibility facade for the central platform tool registry."""

from platform_tools.registry import (
    AgentToolsConfig,
    build_agent_tools,
    parse_agent_tools_config,
)

__all__ = [
    "AgentToolsConfig",
    "build_agent_tools",
    "parse_agent_tools_config",
]
