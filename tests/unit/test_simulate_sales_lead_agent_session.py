from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulate_sales_lead_agent as simulator


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
