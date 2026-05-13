"""Central registry for Phase 3A guarded agent-local tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .logging import GuardrailEventLogger
from .privacy import PrivacyRail
from .scanners import LLMGuardScannerRail
from .tool_policy import (
    ToolPolicyRail,
    ToolSecurityProfile,
    UnprofiledToolPolicy,
    coerce_tool_security_profile,
    read_only_unprofiled_tool_profile,
)


class ToolRegistryError(RuntimeError):
    """Raised when a guarded tool bundle cannot be built safely."""


@dataclass(frozen=True)
class GuardedToolEntry:
    name: str
    tool: Any
    profile: ToolSecurityProfile


@dataclass(frozen=True)
class GuardedToolBundle:
    tools: list[Any]
    middleware: Any
    policy_rail: ToolPolicyRail


class GuardedToolRegistry:
    """Register concrete tools and produce guarded tool bundles for agents."""

    def __init__(
        self,
        *,
        unprofiled_tools: UnprofiledToolPolicy = "block",
    ) -> None:
        self._entries: dict[str, GuardedToolEntry] = {}
        self._unprofiled_tools = unprofiled_tools

    @property
    def entries(self) -> Mapping[str, GuardedToolEntry]:
        return dict(self._entries)

    def register(self, tool: Any, profile: ToolSecurityProfile | Mapping[str, Any]) -> None:
        coerced = coerce_tool_security_profile(profile)
        tool_name = get_tool_name(tool) or coerced.name
        if tool_name != coerced.name:
            raise ToolRegistryError(
                f"Tool name {tool_name!r} does not match security profile {coerced.name!r}."
            )
        if tool_name in self._entries:
            raise ToolRegistryError(f"Duplicate guarded tool name: {tool_name}")
        self._entries[tool_name] = GuardedToolEntry(name=tool_name, tool=tool, profile=coerced)

    def register_unprofiled_read_only(self, tool: Any) -> None:
        if self._unprofiled_tools != "allow_read_only":
            name = get_tool_name(tool) or "<unknown>"
            raise ToolRegistryError(f"Guarded tool {name!r} is missing a security profile.")
        tool_name = get_tool_name(tool)
        if not tool_name:
            raise ToolRegistryError("Unprofiled guarded tool is missing a name.")
        self.register(tool, read_only_unprofiled_tool_profile(tool_name))

    def register_many(
        self,
        tools: Iterable[Any],
        profiles: Mapping[str, ToolSecurityProfile | Mapping[str, Any]],
    ) -> None:
        for tool in tools:
            tool_name = get_tool_name(tool)
            if not tool_name:
                raise ToolRegistryError("Guarded tool is missing a name.")
            profile = profiles.get(tool_name)
            if profile is None:
                self.register_unprofiled_read_only(tool)
            else:
                self.register(tool, profile)

    @classmethod
    def from_tools(
        cls,
        tools: Iterable[Any],
        profiles: Mapping[str, ToolSecurityProfile | Mapping[str, Any]],
        *,
        unprofiled_tools: UnprofiledToolPolicy = "block",
    ) -> "GuardedToolRegistry":
        registry = cls(unprofiled_tools=unprofiled_tools)
        registry.register_many(tools, profiles)
        return registry

    def build_bundle(
        self,
        tool_names: Sequence[str] | None = None,
        *,
        scanner_rail: LLMGuardScannerRail | None = None,
        privacy_rail: PrivacyRail | None = None,
        agent_name: str,
        event_log_path: str | None = None,
        event_logger: GuardrailEventLogger | None = None,
    ) -> GuardedToolBundle:
        selected_names = list(tool_names) if tool_names is not None else list(self._entries)
        missing = [name for name in selected_names if name not in self._entries]
        if missing:
            raise ToolRegistryError(f"Requested guarded tool is not registered: {', '.join(missing)}")

        selected_entries = [self._entries[name] for name in selected_names]
        policy_rail = ToolPolicyRail((entry.profile for entry in selected_entries), unprofiled_tools="block")

        # Imported lazily so policy/registry can be tested without importing the
        # middleware module during package initialization.
        from .middleware import ToolExecutionSafetyMiddleware

        middleware = ToolExecutionSafetyMiddleware(
            policy_rail=policy_rail,
            scanner_rail=scanner_rail,
            privacy_rail=privacy_rail,
            agent_name=agent_name,
            event_log_path=event_log_path,
            event_logger=event_logger,
        )
        return GuardedToolBundle(
            tools=[entry.tool for entry in selected_entries],
            middleware=middleware,
            policy_rail=policy_rail,
        )


def get_tool_name(tool: Any) -> str | None:
    name = getattr(tool, "name", None)
    if name:
        return str(name)
    if isinstance(tool, Mapping):
        mapped_name = tool.get("name")
        if mapped_name:
            return str(mapped_name)
    return None


__all__ = [
    "GuardedToolBundle",
    "GuardedToolEntry",
    "GuardedToolRegistry",
    "ToolRegistryError",
    "get_tool_name",
]
