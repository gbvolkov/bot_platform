from __future__ import annotations

from types import SimpleNamespace

from platform_guardrails.context import build_guardrail_context, privacy_scope_key


def test_build_guardrail_context_uses_configurable_fields():
    context = build_guardrail_context(
        config={
            "configurable": {
                "tenant_id": "tenant-1",
                "user_id": "user-1",
                "user_role": "service_desk",
                "thread_id": "thread-1",
                "model": "base",
                "allow_deanonymization": False,
                "allow_external_search": True,
            }
        },
        agent_name="artifact_creator_agent",
        route="run",
        tool_name="lookup",
        request_id="req-1",
    )

    assert context["tenant_id"] == "tenant-1"
    assert context["user_id"] == "user-1"
    assert context["user_role"] == "service_desk"
    assert context["thread_id"] == "thread-1"
    assert context["agent_name"] == "artifact_creator_agent"
    assert context["route"] == "run"
    assert context["model"] == "base"
    assert context["tool_name"] == "lookup"
    assert context["request_id"] == "req-1"
    assert context["allow_deanonymization"] is False
    assert context["allow_external_search"] is True
    assert privacy_scope_key(context) == "tenant-1|user-1|thread-1"


def test_build_guardrail_context_falls_back_to_runtime_and_stable_scope_parts():
    runtime = SimpleNamespace(
        execution_info=SimpleNamespace(thread_id="runtime-thread"),
        config={},
    )

    context = build_guardrail_context(runtime=runtime, agent_name="sample")

    assert context["tenant_id"] is None
    assert context["user_id"] is None
    assert context["thread_id"] == "runtime-thread"
    assert context["allow_deanonymization"] is True
    assert privacy_scope_key(context) == "__no_tenant__|__no_user__|runtime-thread"
