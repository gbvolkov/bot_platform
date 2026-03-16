import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulate_sysadmin_agent_dialog as simulator


def test_read_text_input_returns_single_line(monkeypatch):
    monkeypatch.setattr(
        simulator,
        "_read_interrupt_input",
        lambda prompt, secret: "docker ps",
    )

    value = simulator._read_text_input(
        "Customer (human)> ",
        secret=False,
        allow_multiline=True,
    )

    assert value == "docker ps"


def test_read_text_input_collects_multiline_block(monkeypatch, capsys):
    values = iter(
        [
            simulator._MULTILINE_START,
            "docker compose ps",
            "docker compose logs --tail=50 web",
            simulator._MULTILINE_END,
        ]
    )
    monkeypatch.setattr(
        simulator,
        "_read_interrupt_input",
        lambda prompt, secret: next(values),
    )

    value = simulator._read_text_input(
        "Customer (human)> ",
        secret=False,
        allow_multiline=True,
    )

    captured = capsys.readouterr()
    assert "[Multiline mode]" in captured.out
    assert value == "docker compose ps\ndocker compose logs --tail=50 web"


def test_handle_command_accepts_multiline_scenario():
    handled, mode, scenario, should_reset = simulator._handle_command(
        "/scenario First line\nSecond line",
        mode="ai",
        scenario="old",
    )

    assert handled is True
    assert mode == "ai"
    assert scenario == "First line\nSecond line"
    assert should_reset is False
