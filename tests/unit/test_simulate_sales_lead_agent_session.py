from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulate_sales_lead_agent as simulator
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_persistent_interactive_session_roundtrip(monkeypatch, tmp_path: Path):
    session_path = tmp_path / "simulator_session.json"
    monkeypatch.setattr(simulator, "_simulator_session_path", lambda: session_path)

    config = {"configurable": {"thread_id": "sales-lead-sim-thread-1"}}
    simulator._save_persistent_interactive_session(config=config, scenario="company_check")

    loaded_config, loaded_scenario, restored = simulator._load_persistent_interactive_session(
        default_scenario="procurement_search"
    )

    assert restored is True
    assert loaded_config == {"configurable": {"thread_id": "sales-lead-sim-thread-1"}}
    assert loaded_scenario == "company_check"
    payload = session_path.read_text(encoding="utf-8")
    assert '"conversation_id": "sales-lead-sim-thread-1"' in payload


def test_persistent_interactive_session_falls_back_to_new_thread(monkeypatch, tmp_path: Path):
    session_path = tmp_path / "simulator_session.json"
    session_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(simulator, "_simulator_session_path", lambda: session_path)

    loaded_config, loaded_scenario, restored = simulator._load_persistent_interactive_session(
        default_scenario="procurement_search"
    )

    assert restored is False
    assert loaded_scenario == "procurement_search"
    assert str(loaded_config["configurable"]["thread_id"]).startswith("sales-lead-sim-")


def test_export_dialog_state_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    export_root = tmp_path / "exports"
    monkeypatch.setattr(simulator, "_simulator_export_root", lambda: export_root)

    class Graph:
        async def aget_state(self, config):
            return SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="Привет"),
                        AIMessage(content="Здравствуйте"),
                    ],
                    "index_id": "sales_lead_permanent",
                }
            )

    config = {"configurable": {"thread_id": "sales-lead-sim-export-1"}}
    export_path, markdown_path = asyncio.run(simulator._export_dialog_state(graph=Graph(), config=config))

    assert export_path.exists()
    assert markdown_path.exists()

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["thread_id"] == "sales-lead-sim-export-1"
    assert payload["message_count"] == 2
    assert payload["state_keys"] == ["index_id", "messages"]
    assert payload["messages"][0]["type"] == "human"
    assert payload["messages"][0]["data"]["content"] == "Привет"
    assert payload["messages"][1]["type"] == "ai"
    assert payload["messages"][1]["data"]["content"] == "Здравствуйте"

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Agent State Export" in markdown
    assert "## 0. Human" in markdown
    assert "## 1. AI" in markdown
    assert "Привет" in markdown
    assert "Здравствуйте" in markdown


def test_export_dialog_state_can_skip_tool_messages(monkeypatch, tmp_path: Path):
    export_root = tmp_path / "exports"
    monkeypatch.setattr(simulator, "_simulator_export_root", lambda: export_root)

    class Graph:
        async def aget_state(self, config):
            return SimpleNamespace(
                values={
                    "messages": [
                        HumanMessage(content="Привет"),
                        AIMessage(content="Здравствуйте"),
                        ToolMessage(content='{"ok": true}', tool_call_id="call-1", name="purchase_lookup_tool"),
                    ],
                    "index_id": "sales_lead_permanent",
                }
            )

    config = {"configurable": {"thread_id": "sales-lead-sim-export-2"}}
    export_path, markdown_path = asyncio.run(
        simulator._export_dialog_state(
            graph=Graph(),
            config=config,
            no_tools=True,
        )
    )

    assert export_path.exists()
    assert markdown_path.exists()
    assert export_path.name.endswith("_state_messages_human_ai.json")

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["thread_id"] == "sales-lead-sim-export-2"
    assert payload["message_filter"] == "human_ai"
    assert payload["message_count"] == 2
    assert [item["type"] for item in payload["messages"]] == ["human", "ai"]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Message filter: human_ai" in markdown
    assert "## 0. Human" in markdown
    assert "## 1. AI" in markdown
    assert "purchase_lookup_tool" not in markdown


def test_read_raw_console_input_prefers_tty_text_input(monkeypatch):
    class TTYStdin:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(simulator.sys, "stdin", TTYStdin())
    monkeypatch.setattr(builtins, "input", lambda prompt: "Привет, мир")

    assert simulator._read_raw_console_input("User> ") == "Привет, мир"
