import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulate_sales_lead_agent as simulator
from agents.utils import ModelType


def test_read_text_input_returns_single_line(monkeypatch):
    monkeypatch.setattr(
        simulator,
        "_read_interrupt_input",
        lambda prompt: "find procurements",
    )

    value = simulator._read_text_input(
        "User> ",
        allow_multiline=True,
    )

    assert value == "find procurements"


def test_read_text_input_collects_multiline_block(monkeypatch, capsys):
    values = iter(
        [
            simulator._MULTILINE_START,
            "first line",
            "second line",
            simulator._MULTILINE_END,
        ]
    )
    monkeypatch.setattr(
        simulator,
        "_read_interrupt_input",
        lambda prompt: next(values),
    )

    value = simulator._read_text_input(
        "User> ",
        allow_multiline=True,
    )

    captured = capsys.readouterr()
    assert "[Multiline mode]" in captured.out
    assert value == "first line\nsecond line"


def test_handle_command_switches_scenario():
    handled, scenario, should_reset = simulator._handle_command(
        "/scenario comparison",
        scenario="procurement_search",
    )

    assert handled is True
    assert scenario == "comparison"
    assert should_reset is False


def test_run_scripted_scenario_reuses_same_thread_and_validates_expectations(tmp_path, monkeypatch):
    calls = []

    class FakeGraph:
        async def ainvoke(self, payload, config):
            thread_id = config["configurable"]["thread_id"]
            calls.append(thread_id)
            if len(calls) == 1:
                return {
                    "messages": [AIMessage(content="Lead list reply")],
                    "normalized_final_answer": {
                        "answer_type": "lead_list",
                        "items": [
                            {
                                "evidence": [
                                    {
                                        "source": "purchase",
                                        "snippet": "Relevant procurement hit",
                                    }
                                ],
                                "fact_statuses": [{"status": "document"}],
                            }
                        ],
                    },
                    "active_run_id": "run-1",
                    "index_id": "index-1",
                    "prepared_documents": [{"document_id": "doc-1", "index_status": "ready", "chunks_count": 2}],
                    "turn_tool_usage": [{"tool": "purchase_search_tool", "status": "success"}],
                }
            return {
                "messages": [AIMessage(content="Fact lookup reply")],
                "normalized_final_answer": {
                    "answer_type": "lead_card",
                    "items": [
                        {
                            "evidence": [
                                {
                                    "source": "document",
                                    "snippet": "Bidder experience requirement",
                                    "file_path": "C:/tmp/doc.pdf",
                                    "page": 3,
                                }
                            ],
                            "fact_statuses": [{"status": "document"}],
                        }
                    ],
                },
                "active_run_id": "run-1",
                "index_id": "index-1",
                "last_doc_search_result": {"index_id": "index-1", "matches": [{"document_id": "doc-1"}]},
                "turn_tool_usage": [{"tool": "doc_search_tool", "status": "success"}],
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    path = asyncio.run(
        simulator.run_scripted_scenario(
            provider=ModelType.GPT,
            scenario="procurement_search_followup_fact_lookup",
        )
    )

    assert path.exists()
    assert len(set(calls)) == 1
    transcript = path.read_text(encoding="utf-8")
    assert "Thread ID" in transcript
    assert "Active Run" in transcript


def test_run_scripted_scenario_reuses_active_run_for_repeated_same_request(tmp_path, monkeypatch):
    calls = []

    class FakeGraph:
        async def ainvoke(self, payload, config):
            thread_id = config["configurable"]["thread_id"]
            calls.append(thread_id)
            return {
                "messages": [AIMessage(content="Lead list reply")],
                "normalized_final_answer": {
                    "answer_type": "lead_list",
                    "items": [
                        {
                            "evidence": [{"source": "purchase", "snippet": "Relevant procurement hit"}],
                            "fact_statuses": [{"status": "document"}],
                        }
                    ],
                },
                "active_run_id": "run-1",
                "index_id": "index-1",
                "prepared_documents": [{"document_id": "doc-1", "index_status": "ready", "chunks_count": 2}],
                "turn_tool_usage": [{"tool": "purchase_search_tool", "status": "success"}],
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    path = asyncio.run(
        simulator.run_scripted_scenario(
            provider=ModelType.GPT,
            scenario="repeated_same_request",
        )
    )

    assert path.exists()
    assert len(calls) == 2
    assert len(set(calls)) == 1
    transcript = path.read_text(encoding="utf-8")
    assert transcript.count("Active Run: `run-1`") == 2


def test_run_scripted_scenario_fails_when_document_evidence_missing(tmp_path, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload, config):
            return {
                "messages": [AIMessage(content="Fact lookup reply")],
                "normalized_final_answer": {"answer_type": "lead_card", "items": []},
                "active_run_id": "run-1",
                "index_id": "index-1",
                "last_doc_search_result": {"index_id": "index-1", "matches": [{"document_id": "doc-1"}]},
                "turn_tool_usage": [{"tool": "doc_search_tool", "status": "success"}],
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    try:
        asyncio.run(
            simulator.run_scripted_scenario(
                provider=ModelType.GPT,
                scenario="fact_lookup",
            )
        )
    except AssertionError as exc:
        assert "document evidence is missing" in str(exc)
    else:
        raise AssertionError("Expected scenario validation to fail.")


def test_run_scripted_scenario_fails_when_required_tool_only_failed(tmp_path, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload, config):
            return {
                "messages": [AIMessage(content="Lead list reply")],
                "normalized_final_answer": {"answer_type": "lead_list", "items": []},
                "active_run_id": "run-1",
                "index_id": "index-1",
                "prepared_documents": [{"document_id": "doc-1", "index_status": "ready", "chunks_count": 2}],
                "turn_tool_usage": [{"tool": "purchase_search_tool", "status": "failed"}],
                "turn_validation": {
                    "issues": [{"code": "purchase_search_failed"}],
                },
                "recommended_next_step": "Retry later.",
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    try:
        asyncio.run(
            simulator.run_scripted_scenario(
                provider=ModelType.GPT,
                scenario="procurement_search",
            )
        )
    except AssertionError as exc:
        assert "required tool purchase_search_tool was not recorded" in str(exc)
    else:
        raise AssertionError("Expected scenario validation to fail.")


def test_run_scripted_scenario_fails_when_partial_answer_has_no_next_step(tmp_path, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload, config):
            return {
                "messages": [AIMessage(content="Lead list reply")],
                "normalized_final_answer": {
                    "answer_type": "lead_list",
                    "items": [],
                    "missing_data": ["procurement_hits"],
                },
                "active_run_id": "run-1",
                "index_id": "index-1",
                "prepared_documents": [{"document_id": "doc-1", "index_status": "ready", "chunks_count": 2}],
                "turn_tool_usage": [{"tool": "purchase_search_tool", "status": "success"}],
                "turn_validation": {"issues": [{"code": "procurement_partial"}]},
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    try:
        asyncio.run(
            simulator.run_scripted_scenario(
                provider=ModelType.GPT,
                scenario="procurement_search",
            )
        )
    except AssertionError as exc:
        assert "recommended_next_step" in str(exc)
    else:
        raise AssertionError("Expected scenario validation to fail.")


def test_run_scripted_scenario_fails_when_searchable_index_support_is_missing(tmp_path, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload, config):
            return {
                "messages": [AIMessage(content="Lead list reply")],
                "normalized_final_answer": {"answer_type": "lead_list", "items": []},
                "active_run_id": "run-1",
                "index_id": "index-1",
                "turn_tool_usage": [{"tool": "purchase_search_tool", "status": "success"}],
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    try:
        asyncio.run(
            simulator.run_scripted_scenario(
                provider=ModelType.GPT,
                scenario="procurement_search",
            )
        )
    except AssertionError as exc:
        assert "searchable prepared index support is missing" in str(exc)
    else:
        raise AssertionError("Expected scenario validation to fail.")


def test_run_scripted_scenario_fails_when_visible_support_missing_for_company_check(tmp_path, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload, config):
            return {
                "messages": [AIMessage(content="Company check reply")],
                "normalized_final_answer": {
                    "answer_type": "company_check",
                    "items": [{"company_name": "Test LLC", "inn": "7707083893"}],
                },
                "turn_tool_usage": [
                    {"tool": "counterparty_scoring_tool", "status": "success"},
                    {"tool": "counterparty_fssp_tool", "status": "success"},
                ],
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    try:
        asyncio.run(
            simulator.run_scripted_scenario(
                provider=ModelType.GPT,
                scenario="company_check",
            )
        )
    except AssertionError as exc:
        assert "visible evidence or fact-status support is missing" in str(exc)
    else:
        raise AssertionError("Expected scenario validation to fail.")


def test_run_scripted_scenario_fails_when_prepared_documents_are_not_searchable(tmp_path, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload, config):
            return {
                "messages": [AIMessage(content="Lead list reply")],
                "normalized_final_answer": {
                    "answer_type": "lead_list",
                    "items": [
                        {
                            "evidence": [{"source": "purchase", "snippet": "hit"}],
                            "fact_statuses": [{"status": "document"}],
                        }
                    ],
                },
                "active_run_id": "run-1",
                "index_id": "index-1",
                "prepared_documents": [{"document_id": "doc-1", "index_status": "failed", "chunks_count": 0}],
                "turn_tool_usage": [{"tool": "purchase_search_tool", "status": "success"}],
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    try:
        asyncio.run(
            simulator.run_scripted_scenario(
                provider=ModelType.GPT,
                scenario="procurement_search",
            )
        )
    except AssertionError as exc:
        assert "searchable prepared index support is missing" in str(exc)
    else:
        raise AssertionError("Expected scenario validation to fail.")


def test_run_scripted_scenario_fails_when_document_fact_status_labels_missing(tmp_path, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload, config):
            return {
                "messages": [AIMessage(content="Fact lookup reply")],
                "normalized_final_answer": {
                    "answer_type": "lead_card",
                    "items": [
                        {
                            "evidence": [
                                {
                                    "source": "document",
                                    "snippet": "Bidder experience requirement",
                                    "file_path": "C:/tmp/doc.pdf",
                                    "page": 3,
                                }
                            ],
                            "fact_statuses": [],
                        }
                    ],
                },
                "active_run_id": "run-1",
                "index_id": "index-1",
                "last_doc_search_result": {"index_id": "index-1", "matches": [{"document_id": "doc-1"}]},
                "turn_tool_usage": [{"tool": "doc_search_tool", "status": "success"}],
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    try:
        asyncio.run(
            simulator.run_scripted_scenario(
                provider=ModelType.GPT,
                scenario="fact_lookup",
            )
        )
    except AssertionError as exc:
        assert "document-backed answer is missing explicit document fact-status labels" in str(exc)
    else:
        raise AssertionError("Expected scenario validation to fail.")


def test_run_scripted_scenario_fails_when_comparison_support_is_missing_for_one_item(tmp_path, monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload, config):
            return {
                "messages": [AIMessage(content="Comparison reply")],
                "normalized_final_answer": {
                    "answer_type": "comparison",
                    "items": [
                        {
                            "company_name": "A",
                            "evidence": [{"source": "scoring", "snippet": "risk low"}],
                            "fact_statuses": [{"status": "external_api"}],
                        },
                        {
                            "company_name": "B",
                            "evidence": [],
                            "fact_statuses": [],
                        },
                    ],
                },
                "turn_tool_usage": [
                    {"tool": "counterparty_scoring_tool", "status": "success"},
                    {"tool": "counterparty_fssp_tool", "status": "success"},
                ],
            }

    monkeypatch.setattr(simulator, "initialize_agent", lambda **kwargs: FakeGraph())
    monkeypatch.setattr(simulator, "_transcript_path", lambda name=None: tmp_path / f"{name}.md")

    try:
        asyncio.run(
            simulator.run_scripted_scenario(
                provider=ModelType.GPT,
                scenario="comparison",
            )
        )
    except AssertionError as exc:
        assert "visible evidence or fact-status support is missing" in str(exc)
    else:
        raise AssertionError("Expected scenario validation to fail.")
