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
    assert '"стоимость владения"' in skill_text
    assert '"руб/км"' in skill_text
    assert "bi-owned ownership-cost attributes" in skill_text
    assert "before `marketing_analyst`, web, a calculated template, or a request for assumptions" in skill_text
    assert "bi request modes" in skill_text
    assert "analytical / comparison mode" in skill_text
    assert "bi does not return a complete model profile in this mode" in skill_text
    assert "do not require bi to return all db fields" in skill_text
    assert "concrete model detail mode" in skill_text
    assert "complete db model profile" in skill_text
    assert "null/empty values as `na`" in skill_text
    assert "mycroft forms the complete model profile" in skill_text
    assert "свесы, габариты?" in skill_text
    assert "specific missing field recovery" in skill_text
    assert "front_overhang_mm" in skill_text


def test_mycroft_tco_skill_requires_bi_lookup_before_template_for_active_models():
    skill_text = (
        Path("skills/mycroft/build-tco-case/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert '"стоимость владения"' in skill_text
    assert '"руб/км"' in skill_text
    assert "call `gaz_pricing_bi_int` first in analytical / selected-field mode" in skill_text
    assert "active target model names" in skill_text
    assert "ownership-cost/tco/cost-per-km fields" in skill_text
    assert "return the value or `na` for each requested bi field" in skill_text
    assert "do not replace this bi ownership-cost lookup with a generic tco template" in skill_text
    assert "after bi has been checked for the active targets" in skill_text
    assert "bi ownership-cost lookup through gaz_pricing_bi_int is mandatory first" in skill_text
    assert "before a template, marketing argument, web estimate, or request for assumptions" in skill_text


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
    assert "bi analytical/comparison output as sufficient only for the fields it returned" in skill_text
    assert "complete db model profile mode" in skill_text
    assert "including null/empty values as `na`" in skill_text
    assert "choose the bi follow-up mode by intent" in skill_text


def test_mycroft_validate_vehicle_facts_rechecks_missing_exact_fields():
    skill_text = (
        Path("skills/mycroft/validate-vehicle-facts/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "do not infer bi absence from prior bi output" in skill_text
    assert "selected fact mode" in skill_text
    assert "not a complete model profile request" in skill_text
    assert "bi does not need to return all db fields" in skill_text
    assert "complete db model profile mode" in skill_text
    assert "complete set of db fields" in skill_text
    assert "null/empty value as `na`" in skill_text
    assert "mycroft will form a complete model profile" in skill_text
    assert "front_overhang_mm" in skill_text
    assert "ask bi to return the value or `na` if the db value is null" in skill_text
    assert "complete db model profile" in skill_text


def test_mycroft_recommendation_skill_uses_analytical_bi_until_full_details_are_requested():
    skill_text = (
        Path("skills/mycroft/build-vehicle-recommendation/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "analytical / comparison mode" in skill_text
    assert "do not require bi to return complete model profiles or all db fields" in skill_text
    assert "selected fields relevant to the recommendation" in skill_text
    assert "complete db model profile mode" in skill_text


def test_mycroft_comparison_skill_does_not_request_complete_profiles_for_analytics():
    skill_text = (
        Path("skills/mycroft/compare-customer-options/SKILL.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "analytical / comparison mode" in skill_text
    assert "do not ask bi for a complete model profile" in skill_text
    assert "do not require bi to return all db fields" in skill_text
    assert "selected comparison columns" in skill_text
    assert "complete db model profile mode" in skill_text


def test_mycroft_other_bi_using_skills_default_to_selected_field_mode():
    skill_paths = [
        Path("skills/mycroft/answer-service-or-operation-question/SKILL.md"),
        Path("skills/mycroft/build-tco-case/SKILL.md"),
        Path("skills/mycroft/handle-competitor-comparison/SKILL.md"),
        Path("skills/mycroft/shortlist-gaz-solutions/SKILL.md"),
        Path("skills/mycroft/prepare-sales-argumentation/SKILL.md"),
        Path("skills/mycroft/prepare-programs-and-financing/SKILL.md"),
        Path("skills/mycroft/email-followup/SKILL.md"),
    ]

    combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in skill_paths)

    assert "selected fact mode" in combined
    assert "analytical / selected-field mode" in combined
    assert "analytical / comparison mode" in combined
    assert "selected-field validation" in combined
    assert "do not request complete db model profiles" in combined
    assert "complete db model profile" in combined


def test_mycroft_bi_service_catalog_defines_analytical_and_complete_profile_modes():
    catalog_text = (
        Path("skills/mycroft/references/subagents-service-catalog.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "bi request modes" in catalog_text
    assert "analytical / comparison services return selected fields" in catalog_text
    assert "they do not return complete model profiles" in catalog_text
    assert "complete db model profile" in catalog_text
    assert "complete set of db fields" in catalog_text
    assert "render every null/empty value as `na`" in catalog_text
    assert "mycroft will form a complete model profile" in catalog_text
    assert "do not ask for all db fields" in catalog_text
    assert "specific missing field recovery" in catalog_text
    assert "ask for value or `na` if the db value is null" in catalog_text
