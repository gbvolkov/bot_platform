"""Policy primitives for guarded tool execution.

Phase 3A intentionally keeps this layer small: profiles describe concrete
agent-local tools, and the execution middleware enforces them at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from .context import GuardrailContext
from .decisions import GuardrailDecision, allow, block, review

ToolSideEffect = Literal["none", "read", "write", "external"]
ToolCategory = Literal[
    "internal_state",
    "retrieval",
    "database",
    "file_export",
    "external_access",
    "notification",
]
ToolSensitivity = Literal["public", "internal", "confidential", "restricted"]
ToolPrivacyTransform = Literal["none", "anonymize", "deanonymize"]
UnprofiledToolPolicy = Literal["block", "allow_read_only"]


@dataclass(frozen=True)
class ToolPrivacyProfile:
    """Privacy transforms to apply around a tool call."""

    argument_transform: ToolPrivacyTransform = "none"
    result_transform: ToolPrivacyTransform = "none"
    transform_command_messages_only: bool = True
    preserve_command_update_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolResultPolicy:
    """Model-visible result minimization policy."""

    scan_result: bool = True
    max_text_chars: int = 12_000
    max_items: int = 30
    allowed_result_keys: tuple[str, ...] = ()
    denied_result_keys: tuple[str, ...] = ()
    sensitive_result_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSecurityProfile:
    """Security policy attached to one concrete tool."""

    name: str
    allowed_roles: tuple[str, ...] = ("default",)
    side_effect: ToolSideEffect = "read"
    category: ToolCategory = "internal_state"
    sensitivity: ToolSensitivity = "internal"
    requires_approval: bool = False
    allow_external_access: bool = False
    allow_file_export: bool = False
    allow_sensitive_data_roles: tuple[str, ...] = ()
    privacy: ToolPrivacyProfile = field(default_factory=ToolPrivacyProfile)
    result_policy: ToolResultPolicy = field(default_factory=ToolResultPolicy)


def coerce_tool_privacy_profile(value: ToolPrivacyProfile | Mapping[str, Any] | None) -> ToolPrivacyProfile:
    if value is None:
        return ToolPrivacyProfile()
    if isinstance(value, ToolPrivacyProfile):
        return value
    return ToolPrivacyProfile(
        argument_transform=value.get("argument_transform", "none"),
        result_transform=value.get("result_transform", "none"),
        transform_command_messages_only=bool(value.get("transform_command_messages_only", True)),
        preserve_command_update_keys=_tuple(value.get("preserve_command_update_keys", ())),
    )


def coerce_tool_result_policy(value: ToolResultPolicy | Mapping[str, Any] | None) -> ToolResultPolicy:
    if value is None:
        return ToolResultPolicy()
    if isinstance(value, ToolResultPolicy):
        return value
    return ToolResultPolicy(
        scan_result=bool(value.get("scan_result", True)),
        max_text_chars=int(value.get("max_text_chars", 12_000)),
        max_items=int(value.get("max_items", 30)),
        allowed_result_keys=_tuple(value.get("allowed_result_keys", ())),
        denied_result_keys=_tuple(value.get("denied_result_keys", ())),
        sensitive_result_keys=_tuple(value.get("sensitive_result_keys", ())),
    )


def coerce_tool_security_profile(value: ToolSecurityProfile | Mapping[str, Any]) -> ToolSecurityProfile:
    if isinstance(value, ToolSecurityProfile):
        return value
    return ToolSecurityProfile(
        name=str(value["name"]),
        allowed_roles=_tuple(value.get("allowed_roles", ("default",))),
        side_effect=value.get("side_effect", "read"),
        category=value.get("category", "internal_state"),
        sensitivity=value.get("sensitivity", "internal"),
        requires_approval=bool(value.get("requires_approval", False)),
        allow_external_access=bool(value.get("allow_external_access", False)),
        allow_file_export=bool(value.get("allow_file_export", False)),
        allow_sensitive_data_roles=_tuple(value.get("allow_sensitive_data_roles", ())),
        privacy=coerce_tool_privacy_profile(value.get("privacy")),
        result_policy=coerce_tool_result_policy(value.get("result_policy")),
    )


def read_only_unprofiled_tool_profile(name: str) -> ToolSecurityProfile:
    """Conservative profile used only when explicitly requested."""

    return ToolSecurityProfile(
        name=name,
        allowed_roles=("default", "service_desk", "sales_manager"),
        side_effect="read",
        category="internal_state",
        sensitivity="internal",
        requires_approval=False,
        allow_external_access=False,
        allow_file_export=False,
        privacy=ToolPrivacyProfile(argument_transform="none", result_transform="anonymize"),
        result_policy=ToolResultPolicy(scan_result=True, max_text_chars=4_000, max_items=20),
    )


class ToolPolicyRail:
    """Evaluate tool security profiles against the current guardrail context."""

    def __init__(
        self,
        profiles: Iterable[ToolSecurityProfile | Mapping[str, Any]],
        *,
        unprofiled_tools: UnprofiledToolPolicy = "block",
    ) -> None:
        self._profiles: dict[str, ToolSecurityProfile] = {}
        self._unprofiled_tools = unprofiled_tools
        for profile in profiles:
            coerced = coerce_tool_security_profile(profile)
            if coerced.name in self._profiles:
                raise ValueError(f"Duplicate tool policy profile: {coerced.name}")
            self._profiles[coerced.name] = coerced

    @property
    def profiles(self) -> Mapping[str, ToolSecurityProfile]:
        return dict(self._profiles)

    def profile_for(self, tool_name: str) -> ToolSecurityProfile | None:
        return self._profiles.get(tool_name)

    def evaluate_call(
        self,
        tool_name: str,
        tool_args: Mapping[str, Any] | None,
        context: GuardrailContext | Mapping[str, Any],
    ) -> GuardrailDecision:
        del tool_args
        profile = self.profile_for(tool_name)
        if profile is None:
            if self._unprofiled_tools == "allow_read_only":
                profile = read_only_unprofiled_tool_profile(tool_name)
            else:
                return block(
                    reason="Tool is not registered with a security profile.",
                    categories=["tool_policy"],
                    metadata={"tool": tool_name},
                )

        role = _context_value(context, "user_role", "default") or "default"
        if profile.allowed_roles and role not in profile.allowed_roles:
            return block(
                reason="Role is not allowed to execute this tool.",
                categories=["tool_policy"],
                metadata={"tool": tool_name, "role": role},
            )

        if profile.requires_approval:
            return review(
                reason="Tool requires approval before execution.",
                categories=["tool_policy"],
                metadata={"tool": tool_name},
            )

        if profile.allow_external_access and not _external_tool_access_allowed(context):
            return block(
                reason="External tool access is disabled for this context.",
                categories=["tool_policy"],
                metadata={"tool": tool_name},
            )

        if profile.allow_file_export and not bool(_context_value(context, "allow_file_export", False)):
            return block(
                reason="File export is disabled for this context.",
                categories=["tool_policy"],
                metadata={"tool": tool_name},
            )

        if profile.sensitivity in {"confidential", "restricted"}:
            has_sensitive_context = bool(_context_value(context, "allow_sensitive_data", False))
            has_sensitive_role = role in profile.allow_sensitive_data_roles
            if not has_sensitive_context and not has_sensitive_role:
                return block(
                    reason="Sensitive data access is disabled for this context.",
                    categories=["tool_policy"],
                    metadata={"tool": tool_name, "sensitivity": profile.sensitivity},
                )

        return allow(
            reason="Tool call is allowed by policy.",
            categories=["tool_policy"],
            metadata={"tool": tool_name},
        )


def minimize_tool_result(value: Any, policy: ToolResultPolicy) -> Any:
    """Return a model-visible value shaped by the result policy."""

    if isinstance(value, str):
        if policy.max_text_chars > 0 and len(value) > policy.max_text_chars:
            return value[: policy.max_text_chars] + "\n[truncated]"
        return value
    if isinstance(value, Mapping):
        allowed = set(policy.allowed_result_keys)
        denied = set(policy.denied_result_keys)
        sensitive = set(policy.sensitive_result_keys)
        minimized: dict[Any, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if allowed and key_text not in allowed:
                continue
            if key_text in denied:
                continue
            if key_text in sensitive:
                minimized[key] = "[redacted]"
                continue
            minimized[key] = minimize_tool_result(item, policy)
        return minimized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        items = list(value)
        if policy.max_items >= 0:
            items = items[: policy.max_items]
        return [minimize_tool_result(item, policy) for item in items]
    return value


ARTIFACT_CREATOR_TOOL_PROFILES: Mapping[str, ToolSecurityProfile] = {
    "commit_artifact_final_text": ToolSecurityProfile(
        name="commit_artifact_final_text",
        allowed_roles=("default", "service_desk", "sales_manager"),
        side_effect="write",
        category="internal_state",
        sensitivity="internal",
        requires_approval=False,
        privacy=ToolPrivacyProfile(
            argument_transform="deanonymize",
            result_transform="anonymize",
            transform_command_messages_only=True,
            preserve_command_update_keys=("artifacts", "phase"),
        ),
        result_policy=ToolResultPolicy(scan_result=True, max_text_chars=4_000, max_items=20),
    )
}


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _context_value(context: GuardrailContext | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(context, key):
        return getattr(context, key)
    if isinstance(context, Mapping):
        return context.get(key, default)
    return default


def _external_tool_access_allowed(context: GuardrailContext | Mapping[str, Any]) -> bool:
    return bool(
        _context_value(
            context,
            "allow_external_tool_access",
            _context_value(context, "allow_external_search", False),
        )
    )


__all__ = [
    "ARTIFACT_CREATOR_TOOL_PROFILES",
    "ToolCategory",
    "ToolPolicyRail",
    "ToolPrivacyProfile",
    "ToolPrivacyTransform",
    "ToolResultPolicy",
    "ToolSecurityProfile",
    "ToolSensitivity",
    "ToolSideEffect",
    "UnprofiledToolPolicy",
    "coerce_tool_security_profile",
    "minimize_tool_result",
    "read_only_unprofiled_tool_profile",
]
