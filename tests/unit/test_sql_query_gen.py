from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

from langchain_core.messages import HumanMessage

from agents import sql_query_gen
from agents.bi_agent import bi_agent


def test_sql_database_max_string_length_can_disable_truncation(tmp_path):
    database_path = tmp_path / "long_text.sqlite"
    long_text = " ".join(f"term_{index}" for index in range(120))
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, body TEXT)")
        connection.execute("INSERT INTO sample (body) VALUES (?)", (long_text,))

    query = "SELECT body FROM sample WHERE id = 1"
    default_db = sql_query_gen.build_sql_database(
        database_url=f"sqlite:///{database_path.as_posix()}"
    )
    default_rows = ast.literal_eval(sql_query_gen.execute_query(query, default_db)["result"])

    full_db = sql_query_gen.build_sql_database(
        database_url=f"sqlite:///{database_path.as_posix()}",
        max_string_length=0,
    )
    full_rows = ast.literal_eval(sql_query_gen.execute_query(query, full_db)["result"])

    assert default_rows[0]["body"].endswith("...")
    assert len(default_rows[0]["body"]) < len(long_text)
    assert full_rows[0]["body"] == long_text


def test_bi_agent_passes_init_context_max_string_length(monkeypatch):
    captured: dict[str, object] = {}

    def fake_get_response(**kwargs):
        captured.update(kwargs)
        return {
            "answer": "ok",
            "query": "SELECT 1",
            "data": "",
            "image": "",
            "graph_type": "",
            "notes": "",
            "row_count": 7,
        }

    monkeypatch.setattr(bi_agent, "get_response", fake_get_response)
    generate_report = bi_agent.create_generate_report_node(
        {
            "database_url": "sqlite:///example.sqlite",
            "return_files": False,
            "return_images": False,
            "max_string_length": 0,
            "answer_row_limit": 50,
        }
    )

    result = generate_report(
        {"messages": [HumanMessage(content="question")]},
        {"configurable": {}},
    )

    assert captured["max_string_length"] == 0
    assert captured["answer_row_limit"] == 50
    assert result["answer"] == "ok"
    assert result["row_count"] == 7


def test_get_response_limits_rows_visible_to_answer_generation(monkeypatch):
    captured: dict[str, object] = {}
    rows = [{"id": index} for index in range(40)]

    monkeypatch.setattr(sql_query_gen, "build_sql_database", lambda **_: object())
    monkeypatch.setattr(sql_query_gen, "write_query", lambda *_, **__: "SELECT id FROM sample")
    monkeypatch.setattr(
        sql_query_gen,
        "execute_query",
        lambda *_: {"result": rows, "error": None},
    )

    def fake_generate_answer(question, query, result, **kwargs):
        captured["result"] = ast.literal_eval(result)
        captured.update(kwargs)
        return {
            "answer": "ok",
            "visual_recommendations": [],
            "measure_columns": [],
            "label_columns": [],
        }

    monkeypatch.setattr(sql_query_gen, "generate_answer", fake_generate_answer)

    response = sql_query_gen.get_response(
        "show sample",
        database_url="sqlite:///example.sqlite",
        return_files=False,
        return_images=False,
        answer_row_limit=5,
    )

    assert response["answer"] == "ok"
    assert len(captured["result"]) == 5
    assert captured["sql_row_count"] == 40
    assert captured["answer_row_count"] == 5
    assert captured["result_truncated"] is True
    assert response["row_count"] == 40
    assert response["notes"] == "Only showing top 5 rows of 40."


def test_get_response_zero_uses_default_answer_row_limit(monkeypatch):
    captured: dict[str, object] = {}
    rows = [{"id": index} for index in range(40)]

    monkeypatch.setattr(sql_query_gen, "build_sql_database", lambda **_: object())
    monkeypatch.setattr(sql_query_gen, "write_query", lambda *_, **__: "SELECT id FROM sample")
    monkeypatch.setattr(
        sql_query_gen,
        "execute_query",
        lambda *_: {"result": rows, "error": None},
    )

    def fake_generate_answer(question, query, result, **kwargs):
        captured["result"] = ast.literal_eval(result)
        captured.update(kwargs)
        return {
            "answer": "ok",
            "visual_recommendations": [],
            "measure_columns": [],
            "label_columns": [],
        }

    monkeypatch.setattr(sql_query_gen, "generate_answer", fake_generate_answer)

    response = sql_query_gen.get_response(
        "show sample",
        database_url="sqlite:///example.sqlite",
        return_files=False,
        return_images=False,
        answer_row_limit=0,
    )

    assert len(captured["result"]) == 30
    assert captured["sql_row_count"] == 40
    assert captured["answer_row_count"] == 30
    assert captured["result_truncated"] is True
    assert response["row_count"] == 40
    assert response["notes"] == "Only showing top 30 rows of 40."


def test_get_response_negative_answer_row_limit_disables_limit(monkeypatch):
    captured: dict[str, object] = {}
    rows = [{"id": index} for index in range(40)]

    monkeypatch.setattr(sql_query_gen, "build_sql_database", lambda **_: object())
    monkeypatch.setattr(sql_query_gen, "write_query", lambda *_, **__: "SELECT id FROM sample")
    monkeypatch.setattr(
        sql_query_gen,
        "execute_query",
        lambda *_: {"result": rows, "error": None},
    )

    def fake_generate_answer(question, query, result, **kwargs):
        captured["result"] = ast.literal_eval(result)
        captured.update(kwargs)
        return {
            "answer": "ok",
            "visual_recommendations": [],
            "measure_columns": [],
            "label_columns": [],
        }

    monkeypatch.setattr(sql_query_gen, "generate_answer", fake_generate_answer)

    response = sql_query_gen.get_response(
        "show sample",
        database_url="sqlite:///example.sqlite",
        return_files=False,
        return_images=False,
        answer_row_limit=-1,
    )

    assert len(captured["result"]) == 40
    assert captured["sql_row_count"] == 40
    assert captured["answer_row_count"] == 40
    assert captured["result_truncated"] is False
    assert response["row_count"] == 40
    assert response["notes"] == ""


def test_kpi_bi_int_configures_bi_result_handling():
    load_config = json.loads(
        Path("data/config/bot_service/load.json").read_text(encoding="utf-8")
    )
    kpi_agent = next(
        agent for agent in load_config["agents"] if agent["id"] == "kpi_bi_int"
    )

    assert kpi_agent["init_context"]["max_string_length"] == 0
    assert kpi_agent["init_context"]["answer_row_limit"] == 50
