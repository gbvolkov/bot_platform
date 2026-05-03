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
        "answer-service-or-operation-question",
        "answer-synthesis",
        "artifact-export",
        "build-tco-case",
        "build-vehicle-recommendation",
        "capture-customer-requirements",
        "compare-customer-options",
        "email-followup",
        "handle-competitor-comparison",
        "prepare-programs-and-financing",
        "prepare-sales-argumentation",
        "shortlist-gaz-solutions",
        "source-authority-and-routing",
        "validate-vehicle-facts",
    }


def test_mycroft_routing_skill_defines_source_capability_boundaries():
    skill_text = (
        Path("skills/mycroft/source-authority-and-routing/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "broad fit or initial candidate discovery" in skill_text
    assert "formal filter parameters suitable for bi" in skill_text
    assert "do not ask bi to" in skill_text
    assert "create questionnaires" in skill_text
    assert "plan the sales dialogue" in skill_text
    assert "preserve the user's scope" in skill_text
    assert "gaz-only; exclude non-gaz and competitor models" in skill_text
    assert "if one source returns a gap, use the next appropriate source" in skill_text
    assert "exact vehicle dimensions and geometry" in skill_text
    assert "a concrete bi target may be composed from" in skill_text
    assert "that exact attribute is not already present in the visible context" in skill_text
    assert "do not infer bi absence from a previous bi answer" in skill_text
    assert "complete non-duplicate model field profile" in skill_text
    assert "exclude `_nocase` mirror fields" in skill_text
    assert "свесы, габариты?" in skill_text
    assert 'not a curated "important fields" subset' in skill_text
    assert "specific missing field recovery" in skill_text
    assert "front_overhang_mm" in skill_text


def test_mycroft_answer_synthesis_skill_requires_latest_mix_consistency():
    skill_text = (
        Path("skills/mycroft/answer-synthesis/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "preserve the latest active recommendation" in skill_text
    assert "do not revert to an older preliminary mix" in skill_text
    assert "a candidate from marketing is missing bi confirmation" in skill_text
    assert "a fleet split repeats the same model/modification" in skill_text
    assert "for the exact requested attributes" in skill_text
    assert "previous bi output may be reused only for the exact fields it returned" in skill_text
    assert "complete non-duplicate model profile" in skill_text
    assert "only a narrow previous lookup" in skill_text
    assert "specific missing field recovery" in skill_text
    assert "that specific field and aliases" in skill_text


def test_mycroft_validate_vehicle_facts_rechecks_missing_exact_fields():
    skill_text = (
        Path("skills/mycroft/validate-vehicle-facts/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "do not infer bi absence from prior bi output" in skill_text
    assert "request a complete non-duplicate model field profile" in skill_text
    assert "even when the user asks for one field, two fields, or shorthand" in skill_text
    assert "exclude duplicate mirror fields" in skill_text
    assert 'not a curated "important fields" subset' in skill_text
    assert "minimum checklist" in skill_text
    assert "targeted bi follow-up" in skill_text
    assert "front_overhang_mm" in skill_text
    assert "before asking the user to clarify units" in skill_text
    assert "complete model field profile" in skill_text
    assert "specific missing field recovery" in skill_text


def test_mycroft_recommendation_skill_requests_complete_bi_profiles_and_recovery():
    skill_text = (
        Path("skills/mycroft/build-vehicle-recommendation/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "request a complete non-duplicate model field profile for each candidate" in skill_text
    assert "not a curated list of important fields" in skill_text
    assert "every user-facing original bi field" in skill_text
    assert "minimum checklist" in skill_text
    assert "specific missing field recovery" in skill_text
    assert "targeted bi follow-up" in skill_text


def test_mycroft_comparison_skill_requests_complete_bi_profiles_and_recovery():
    skill_text = (
        Path("skills/mycroft/compare-customer-options/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "request a complete non-duplicate model field profile for each option" in skill_text
    assert 'not a curated "important fields" subset' in skill_text
    assert "every user-facing original bi field" in skill_text
    assert "minimum checklist" in skill_text
    assert "specific missing field recovery" in skill_text
    assert "targeted bi follow-up" in skill_text


def test_mycroft_bi_service_catalog_defines_complete_model_profile():
    catalog_text = (
        Path("skills/mycroft/references/subagents-service-catalog.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "complete model field profile" in catalog_text
    assert "returns every user-facing original bi field" in catalog_text
    assert 'not a curated "important fields" subset' in catalog_text
    assert "`_nocase` mirror fields" in catalog_text
    assert "raw/import/source technical columns" in catalog_text
    assert "specific missing field recovery" in catalog_text
    assert "does not ask the user to clarify units before checking bi" in catalog_text
