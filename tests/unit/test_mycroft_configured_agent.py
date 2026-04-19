from __future__ import annotations

import json
from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import _list_skills

from agents.mycroft_agent.configured_agent import build_skills_backend, initialize_agent
from agents.utils import ModelType


def test_configured_agent_builds_mycroft_from_config(monkeypatch, tmp_path):
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text("Configured Mycroft prompt.", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "system_prompt": {"type": "file", "path": "system_prompt.txt"},
                "skills": {"paths": ["skills/marketing_analyst"]},
                "subagents": {
                    "stateless": ["stateless_agent"],
                    "stateful": ["stateful_agent"],
                },
                "internal_tools": ["store_artifact_tool"],
                "mcp": {"tool_name_prefix": True, "servers": []},
                "deepagents": {
                    "interrupt_on": {
                        "send_message": {
                            "allowed_decisions": ["approve"],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    async def fake_initialize_subagents(agent_ids):
        return [{"name": agent_id, "description": agent_id, "runnable": object()} for agent_id in agent_ids]

    async def fake_load_mcp_tools(_mcp_config):
        return ["mcp_tool"]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "agents.mycroft_agent.configured_agent.initialize_configured_subagents",
        fake_initialize_subagents,
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.configured_agent.build_internal_tools",
        lambda specs: ["internal_tool"],
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.configured_agent.load_mcp_tools_from_config",
        fake_load_mcp_tools,
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.configured_agent.initialize_mycroft_agent",
        lambda **kwargs: captured.update(kwargs) or {"agent": kwargs},
    )

    result = initialize_agent(
        provider=ModelType.GPT,
        config_path=config_path,
        model_size="mini",
        checkpoint_saver="checkpoint",
    )

    assert result["agent"]["system_prompt"] == "Configured Mycroft prompt."
    assert captured["model_size"] == "mini"
    assert captured["tools"] == ["internal_tool", "mcp_tool"]
    assert [agent["name"] for agent in captured["stateless_subagents"]] == ["stateless_agent"]
    assert [agent["name"] for agent in captured["stateful_subagents"]] == ["stateful_agent"]
    assert captured["checkpoint_saver"] == "checkpoint"
    assert captured["interrupt_on"] == {"send_message": {"allowed_decisions": ["approve"]}}
    assert captured["skills"] == ("/skills/marketing_analyst",)
    assert isinstance(captured["backend"], FilesystemBackend)


def test_configured_agent_accepts_repo_relative_config_path(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_initialize_subagents(_agent_ids):
        return []

    async def fake_load_mcp_tools(_mcp_config):
        return []

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "agents.mycroft_agent.configured_agent.initialize_configured_subagents",
        fake_initialize_subagents,
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.configured_agent.build_internal_tools",
        lambda specs: [],
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.configured_agent.load_mcp_tools_from_config",
        fake_load_mcp_tools,
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.configured_agent.initialize_mycroft_agent",
        lambda **kwargs: captured.update(kwargs) or {"agent": kwargs},
    )

    initialize_agent(
        provider=ModelType.GPT,
        config_path="agents/mycroft_agent/scenarios/marketing_analyst/config.json",
    )

    assert "internal GAZ marketing-materials analyst" in captured["system_prompt"]
    assert captured["skills"] == ("/skills/marketing_analyst",)


def test_build_skills_backend_loads_virtual_skills_on_windows_paths():
    backend = build_skills_backend(("/skills/marketing_analyst",))

    skills = _list_skills(backend, "/skills/marketing_analyst")

    assert {skill["name"] for skill in skills} == {
        "claims-guardrails",
        "comparison-workflow",
        "evidence-packaging",
        "landscape-workflow",
        "source-policy",
    }


def test_build_skills_backend_loads_mycroft_orchestrator_skills():
    backend = build_skills_backend(("/skills/mycroft",))

    skills = _list_skills(backend, "/skills/mycroft")

    assert {skill["name"] for skill in skills} == {
        "answer-synthesis",
        "artifact-export",
        "consultative-sales-dialogue",
        "dealer-service-lookup",
        "email-followup",
        "source-authority-and-routing",
        "vin-and-recall",
    }


def test_mycroft_routing_skill_defines_source_capability_boundaries():
    skill_text = (
        Path("skills/mycroft/source-authority-and-routing/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "broad model and family overviews" in skill_text
    assert "model search/filtering by concrete operational parameters" in skill_text
    assert "do not ask bi to design questionnaires" in skill_text
    assert "field lists" in skill_text
    assert "calculation methodology" in skill_text
    assert "preserve the user's brand/scope constraint" in skill_text
    assert "gaz models only" in skill_text
    assert "gaz-only; exclude non-gaz and competitor models" in skill_text
    assert "do not relabel out-of-scope models" in skill_text
    assert "do not stop at \"i can search again\"" in skill_text
    assert "a follow-up `gaz_pricing_bi_int` request for those candidates is required" in skill_text
    assert "make an additional `gaz_pricing_bi_int` request" in skill_text
    assert "do not seed `gaz_pricing_bi_int` with only mycroft's guessed models" in skill_text
    assert "do not include example models" in skill_text
    assert "never use the `general-purpose` subagent" in skill_text


def test_mycroft_answer_synthesis_skill_requires_latest_mix_consistency():
    skill_text = (
        Path("skills/mycroft/answer-synthesis/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "exact latest mix first" in skill_text
    assert "check the conversation history before agreeing" in skill_text
    assert "generic checklist" in skill_text
    assert "candidate lacks bi confirmation" in skill_text
    assert "do not present a fleet ratio that repeats the same model/modification" in skill_text
