from __future__ import annotations

import json
import asyncio
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage

import artifact_creator_agent_cli as cli
from agents.utils import ModelType


class FakeAgent:
    def __init__(self) -> None:
        self.inputs: list[dict] = []
        self.configs: list[dict] = []

    async def ainvoke(self, input_value, config=None):
        self.inputs.append(input_value)
        self.configs.append(config)
        return {"messages": [AIMessage(content="assistant answer")]}


def test_parse_provider_accepts_model_type_value_and_name():
    assert cli._parse_provider("openai") == ModelType.GPT
    assert cli._parse_provider("GPT") == ModelType.GPT


def test_parse_scanner_failure_policy_rejects_unknown_value():
    assert cli._parse_scanner_failure_policy("fail_open") == "fail_open"

    try:
        cli._parse_scanner_failure_policy("bad")
    except ValueError as exc:
        assert "fail_closed or fail_open" in str(exc)
    else:
        raise AssertionError("Expected scanner failure policy validation to fail.")


def test_new_config_includes_guardrail_scope_fields():
    thread_id, run_config = cli._new_config(
        "thread-123",
        user_id="user-1",
        user_role="reviewer",
        tenant_id="tenant-1",
    )

    assert thread_id == "thread-123"
    assert run_config == {
        "configurable": {
            "thread_id": "thread-123",
            "tenant_id": "tenant-1",
            "user_id": "user-1",
            "user_role": "reviewer",
        }
    }


def test_new_config_can_enable_external_tool_access_context():
    _thread_id, run_config = cli._new_config(
        "thread-123",
        allow_external_tool_access=True,
    )

    assert run_config["configurable"]["allow_external_tool_access"] is True


def test_new_config_accepts_deprecated_external_search_alias():
    _thread_id, run_config = cli._new_config(
        "thread-123",
        allow_external_search=True,
    )

    assert run_config["configurable"]["allow_external_tool_access"] is True


def test_build_human_message_assigns_id_and_text_segment():
    message = cli._build_human_message("hello")

    assert message.id
    assert message.id.startswith("human-")
    assert message.content == [{"type": "text", "text": "hello"}]


def test_invoke_turn_sends_human_message_with_id():
    agent = FakeAgent()
    run_config = {"configurable": {"thread_id": "thread"}}

    answer = asyncio.run(cli._invoke_turn(agent, user_text="hello", config=run_config))

    assert answer == "assistant answer"
    assert agent.configs == [run_config]
    messages = agent.inputs[0]["messages"]
    assert len(messages) == 1
    assert messages[0].id.startswith("human-")
    assert messages[0].content == [{"type": "text", "text": "hello"}]


def test_save_thread_export_writes_json(tmp_path):
    path = tmp_path / "thread.json"

    saved_path = cli._save_thread_export(
        path=path,
        thread_id="thread-1",
        provider="openai",
        guardrails_enabled=True,
        scanners_enabled=None,
        turns=[{"user": "u", "assistant": "a"}],
    )

    assert saved_path == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["thread_id"] == "thread-1"
    assert payload["guardrails_enabled"] is True
    assert payload["scanners_enabled"] is None
    assert payload["turns"] == [{"user": "u", "assistant": "a"}]


def test_resolve_system_prompt_path_prefers_prompts_dir(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_path = prompts_dir / "artifact.txt"
    prompt_path.write_text("prompt", encoding="utf-8")
    monkeypatch.setattr(cli, "DEFAULT_PROMPTS_DIR", prompts_dir)

    assert cli._resolve_system_prompt_path("artifact.txt") == prompt_path


def test_load_agent_registry_settings_reads_tools_and_guardrail_profiles(tmp_path):
    config_path = tmp_path / "load.json"
    prompt_model_config = {
        "path": "custom/prompt-injection-model",
        "kwargs": {
            "id2label": {"0": "SAFE", "1": "INJECTION"},
            "label2id": {"SAFE": 0, "INJECTION": 1},
        },
        "pipeline_kwargs": {"max_length": 256, "truncation": True},
        "tokenizer_kwargs": {"extra_special_tokens": {}},
    }
    config_path.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "id": "artifact_creator_agent",
                        "tools": [
                            {"type": "internal", "name": "web_search_tool"},
                            {
                                "type": "mcp",
                                "server": "googleapis_maps",
                                "tools": ["search_places"],
                            },
                        ],
                        "params": {
                            "guardrail_tool_profiles": {
                                "web_search": {
                                    "name": "web_search",
                                    "allowed_roles": ["default"],
                                    "side_effect": "read",
                                    "category": "external_access",
                                }
                            },
                            "guardrail_unprofiled_tools": "allow_read_only",
                            "guardrail_prompt_injection_model": prompt_model_config,
                            "guardrail_prompt_injection_threshold": 0.5,
                            "guardrail_palimpsest_run_entities": ["RU_PERSON"],
                            "guardrail_palimpsest_entity_table": {
                                "RU_PERSON": {"placeholder": "PERSON"}
                            },
                            "guardrail_palimpsest_typed_placeholders": True,
                            "guardrail_palimpsest_options": {"placeholder_mode": "typed"},
                            "guardrail_palimpsest_session_options": {"placeholder_style": "typed"},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    settings = cli._load_agent_registry_settings(config_path)

    assert settings.tools_config.internal_tools[0].name == "web_search_tool"
    assert settings.tools_config.mcp.servers[0].name == "maps"
    assert settings.tools_config.mcp.servers[0].tools == ("search_places",)
    assert settings.guardrail_tool_profiles["web_search"]["allowed_roles"] == ["default"]
    assert settings.guardrail_unprofiled_tools == "allow_read_only"
    assert settings.guardrail_prompt_injection_model == prompt_model_config
    assert settings.guardrail_prompt_injection_model_revision is None
    assert settings.guardrail_prompt_injection_threshold == 0.5
    assert settings.guardrail_palimpsest_run_entities == ["RU_PERSON"]
    assert settings.guardrail_palimpsest_entity_table == {
        "RU_PERSON": {"placeholder": "PERSON"}
    }
    assert settings.guardrail_palimpsest_typed_placeholders is True
    assert settings.guardrail_palimpsest_options == {"placeholder_mode": "typed"}
    assert settings.guardrail_palimpsest_session_options == {"placeholder_style": "typed"}


def test_build_registry_tools_delegates_to_platform_tool_registry(monkeypatch):
    captured = {}
    expected_tools = [SimpleNamespace(name="web_search")]
    settings = cli.ArtifactAgentRegistrySettings(
        tools_config=cli.parse_agent_tools_config(["web_search_tool"], agent_id="artifact_creator_agent"),
        guardrail_tool_profiles={},
        guardrail_unprofiled_tools="block",
    )

    async def fake_build_agent_tools(tools_config):
        captured["tools_config"] = tools_config
        return expected_tools

    monkeypatch.setattr(cli, "build_agent_tools", fake_build_agent_tools)

    assert asyncio.run(cli._build_registry_tools(settings)) == expected_tools
    assert captured["tools_config"] is settings.tools_config


def test_normalize_export_path_adds_markdown_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    path = cli._normalize_export_path("saved-thread", thread_id="thread-1")

    assert path == Path(tmp_path / "saved-thread.md")
