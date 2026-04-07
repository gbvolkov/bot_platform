from __future__ import annotations

import pytest

from agents.mycroft_agent import web_search_subagent
from agents.mycroft_agent.web_search_subagent import (
    WEB_SEARCH_AGENT_DESCRIPTION,
    WEB_SEARCH_AGENT_ID,
    WEB_SEARCH_AGENT_SYSTEM_PROMPT,
)


class FakeYandexSearchTool:
    name = "web_search"

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_web_search_subagent_returns_dictionary_subagent(monkeypatch):
    monkeypatch.setattr(web_search_subagent.config, "YA_API_KEY", "ya-key")
    monkeypatch.setattr(web_search_subagent.config, "YA_FOLDER_ID", "folder-id")
    monkeypatch.setattr(web_search_subagent, "YandexSearchTool", FakeYandexSearchTool)

    subagent = web_search_subagent.build_web_search_subagent(
        max_results=7,
        summarize=False,
    )

    assert subagent["name"] == WEB_SEARCH_AGENT_ID
    assert subagent["description"] == WEB_SEARCH_AGENT_DESCRIPTION
    assert subagent["system_prompt"] == WEB_SEARCH_AGENT_SYSTEM_PROMPT
    assert len(subagent["tools"]) == 1
    tool = subagent["tools"][0]
    assert isinstance(tool, FakeYandexSearchTool)
    assert tool.kwargs == {
        "api_key": "ya-key",
        "folder_id": "folder-id",
        "max_results": 7,
        "summarize": False,
    }


def test_build_web_search_tool_requires_yandex_environment(monkeypatch):
    monkeypatch.setattr(web_search_subagent.config, "YA_API_KEY", "")
    monkeypatch.setattr(web_search_subagent.config, "YA_FOLDER_ID", "")

    with pytest.raises(ValueError) as exc_info:
        web_search_subagent.build_web_search_tool()

    assert "YA_API_KEY" in str(exc_info.value)
    assert "YA_FOLDER_ID" in str(exc_info.value)
