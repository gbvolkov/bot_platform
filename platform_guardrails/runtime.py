from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import resolve_guardrail_policy
from .middleware import (
    PrivacyModelRequestMiddleware,
    SecurityScannerMiddleware,
    ToolExecutionSafetyMiddleware,
)
from .privacy import PrivacyRail
from .scanners import LLMGuardScannerProfile, LLMGuardScannerRail
from .tool_policy import ToolSecurityProfile, coerce_tool_security_profile
from .tool_registry import GuardedToolRegistry


def _truthy(value: Any) -> bool:
    return bool(value)


@dataclass
class PlatformGuardrailRuntime:
    agent_id: str
    policy_id: str | None = None
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    event_log_path: str | None = None
    scanner_rail: LLMGuardScannerRail | None = None
    privacy_rail: PrivacyRail | None = None

    @classmethod
    def disabled(cls, *, agent_id: str) -> "PlatformGuardrailRuntime":
        return cls(agent_id=agent_id)

    @classmethod
    def from_policy_id(
        cls,
        policy_id: str | None,
        *,
        agent_id: str,
    ) -> "PlatformGuardrailRuntime":
        if not policy_id:
            return cls.disabled(agent_id=agent_id)
        return cls.from_policy_kwargs(
            resolve_guardrail_policy(policy_id),
            agent_id=agent_id,
            policy_id=policy_id,
        )

    @classmethod
    def from_policy_kwargs(
        cls,
        policy_kwargs: Mapping[str, Any],
        *,
        agent_id: str,
        policy_id: str | None = None,
    ) -> "PlatformGuardrailRuntime":
        kwargs = dict(policy_kwargs)
        event_log_path = f"./logs/{agent_id}_{time.strftime('%Y%m%d%H%M')}_guardrails.jsonl"
        scanner_rail = None
        privacy_rail = None

        if _truthy(kwargs.get("guardrail_privacy_enabled")):
            privacy_rail = _build_privacy_rail(kwargs)

        if _truthy(kwargs.get("guardrail_scanners_enabled")):
            scanner_rail = _build_scanner_rail(kwargs)

        return cls(
            agent_id=agent_id,
            policy_id=policy_id,
            policy_kwargs=kwargs,
            event_log_path=event_log_path,
            scanner_rail=scanner_rail,
            privacy_rail=privacy_rail,
        )

    @property
    def privacy_enabled(self) -> bool:
        return self.privacy_rail is not None

    @property
    def scanners_enabled(self) -> bool:
        return self.scanner_rail is not None

    @property
    def tool_execution_enabled(self) -> bool:
        return bool(self.policy_kwargs.get("guardrail_tool_execution_enabled", False))

    @property
    def composite_input_scanners(self) -> tuple[str, ...] | None:
        value = self.policy_kwargs.get("guardrail_composite_input_scanners")
        return None if value is None else tuple(value)

    @property
    def composite_recent_message_limit(self) -> int:
        return int(self.policy_kwargs.get("guardrail_composite_recent_message_limit", 20))

    @property
    def scan_system_prompt(self) -> bool:
        return bool(self.policy_kwargs.get("guardrail_scan_system_prompt", True))

    @property
    def verbose_logging(self) -> bool:
        return bool(self.policy_kwargs.get("guardrail_verbose_logging", False))

    def state_keys_for_policy(self, scan_state_keys: tuple[str, ...]) -> tuple[str, ...]:
        if self.scan_system_prompt:
            return tuple(scan_state_keys)
        return tuple(key for key in scan_state_keys if key != "system_prompt")

    def security_middleware(
        self,
        *,
        agent_name: str,
        scan_system_prompt: bool = False,
        scan_state_keys: tuple[str, ...] = (),
        composite_input_scanners: tuple[str, ...] | None = None,
        composite_recent_message_limit: int | None = None,
        composite_message_roles: tuple[str, ...] | None = None,
    ) -> SecurityScannerMiddleware | None:
        if self.scanner_rail is None:
            return None
        return SecurityScannerMiddleware(
            self.scanner_rail,
            agent_name=agent_name,
            event_log_path=self.event_log_path,
            scan_system_prompt=scan_system_prompt and self.scan_system_prompt,
            scan_state_keys=self.state_keys_for_policy(scan_state_keys),
            include_system_prompt_in_scans=self.scan_system_prompt,
            composite_input_scanners=(
                self.composite_input_scanners
                if composite_input_scanners is None
                else composite_input_scanners
            ),
            composite_recent_message_limit=(
                self.composite_recent_message_limit
                if composite_recent_message_limit is None
                else composite_recent_message_limit
            ),
            composite_message_roles=composite_message_roles,
        )

    def privacy_middleware(
        self,
        *,
        agent_name: str,
        guard_tool_calls: bool = True,
    ) -> PrivacyModelRequestMiddleware | None:
        if self.privacy_rail is None:
            return None
        return PrivacyModelRequestMiddleware(
            self.privacy_rail,
            agent_name=agent_name,
            guard_tool_calls=guard_tool_calls,
            event_log_path=self.event_log_path,
        )

    def tool_execution_middleware(
        self,
        *,
        tools: list[Any],
        tool_profiles: Mapping[str, ToolSecurityProfile | Mapping[str, Any]],
        agent_name: str,
    ) -> ToolExecutionSafetyMiddleware | None:
        if not self.tool_execution_enabled:
            return None
        profiles = dict(tool_profiles)
        _validate_tool_runtime_dependencies(
            profiles,
            scanners_enabled=self.scanners_enabled,
            privacy_enabled=self.privacy_enabled,
        )
        registry = GuardedToolRegistry(
            unprofiled_tools=self.policy_kwargs.get("guardrail_unprofiled_tools", "block")
        )
        registry.register_many(tools, profiles)
        bundle = registry.build_bundle(
            agent_name=agent_name,
            scanner_rail=self.scanner_rail,
            privacy_rail=self.privacy_rail,
            event_log_path=self.event_log_path,
        )
        return bundle.middleware


def _build_privacy_rail(kwargs: Mapping[str, Any]) -> PrivacyRail:
    palimpsest_kwargs: dict[str, Any] = {
        "locale": kwargs.get("guardrails_locale", "ru-RU"),
    }
    if kwargs.get("guardrail_palimpsest_run_entities") is not None:
        palimpsest_kwargs["run_entities"] = kwargs["guardrail_palimpsest_run_entities"]
    if kwargs.get("guardrail_palimpsest_entity_replacements") is not None:
        palimpsest_kwargs["entity_replacements"] = kwargs["guardrail_palimpsest_entity_replacements"]
    if kwargs.get("guardrail_palimpsest_options") is not None:
        palimpsest_kwargs["palimpsest_options"] = kwargs["guardrail_palimpsest_options"]
    if kwargs.get("guardrail_palimpsest_session_options") is not None:
        palimpsest_kwargs["palimpsest_session_options"] = kwargs["guardrail_palimpsest_session_options"]
    return PrivacyRail.from_palimpsest(**palimpsest_kwargs)


def _build_scanner_rail(kwargs: Mapping[str, Any]) -> LLMGuardScannerRail:
    scanner_profile_kwargs: dict[str, Any] = {
        "banned_topics": kwargs.get("guardrail_banned_topics"),
        "failure_policy": kwargs.get("guardrail_scanner_failure_policy", "fail_closed"),
    }
    if kwargs.get("guardrail_prompt_injection_model") is not None:
        scanner_profile_kwargs["prompt_injection_model"] = kwargs["guardrail_prompt_injection_model"]
    if kwargs.get("guardrail_prompt_injection_model_revision") is not None:
        scanner_profile_kwargs["prompt_injection_model_revision"] = kwargs[
            "guardrail_prompt_injection_model_revision"
        ]
    if kwargs.get("guardrail_prompt_injection_threshold") is not None:
        scanner_profile_kwargs["prompt_injection_threshold"] = kwargs[
            "guardrail_prompt_injection_threshold"
        ]
    if kwargs.get("guardrail_tool_result_prompt_injection_threshold") is not None:
        scanner_profile_kwargs["tool_result_prompt_injection_threshold"] = kwargs[
            "guardrail_tool_result_prompt_injection_threshold"
        ]
    if kwargs.get("guardrail_url_policy") is not None:
        scanner_profile_kwargs["url_policy"] = kwargs["guardrail_url_policy"]
    return LLMGuardScannerRail(
        LLMGuardScannerProfile.artifact_creator_default(**scanner_profile_kwargs),
        verbose_logging=bool(kwargs.get("guardrail_verbose_logging", False)),
    )


def _validate_tool_runtime_dependencies(
    profiles: Mapping[str, ToolSecurityProfile | Mapping[str, Any]],
    *,
    scanners_enabled: bool,
    privacy_enabled: bool,
) -> None:
    for raw_profile in profiles.values():
        profile = coerce_tool_security_profile(raw_profile)
        if profile.result_policy.scan_result and not scanners_enabled:
            raise RuntimeError(
                "Tool execution guardrails selected a profile with result_policy.scan_result=true, "
                "but scanner guardrails are disabled."
            )
        if profile.privacy.result_transform != "none" and not privacy_enabled:
            raise RuntimeError(
                "Tool execution guardrails selected a profile with privacy transforms, "
                "but privacy guardrails are disabled."
            )


__all__ = ["PlatformGuardrailRuntime"]
