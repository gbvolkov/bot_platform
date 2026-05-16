from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler

from agents.new_theodor_agent import agent as theodor_agent
from agents.new_theodor_agent import artifacts_defs
from agents.utils import ModelType


def _artifact(artifact_id: int, name: str) -> dict:
    return {
        "id": artifact_id,
        "stage": "stage",
        "stage_goal": "stage goal",
        "name": name,
        "goal": "goal",
        "components": [],
        "methodology": "methodology",
        "criteria": [],
    }


def test_artifact_helpers_handle_sparse_ids():
    original_artifacts = list(artifacts_defs.ARTIFACTS)
    try:
        artifacts_defs.ARTIFACTS.clear()
        artifacts_defs.ARTIFACTS.extend(
            [
                _artifact(0, "First"),
                _artifact(1, "Second"),
                _artifact(5, "Third"),
                _artifact(12, "Fourth"),
            ]
        )

        assert artifacts_defs.normalize_artifact_id(2) == 5
        assert artifacts_defs.normalize_artifact_id(3) == 12
        assert artifacts_defs.get_next_artifact_id(1) == 5
        assert artifacts_defs.get_next_artifact_id(5) == 12
        assert artifacts_defs.get_next_artifact_id(12) is None
    finally:
        artifacts_defs.ARTIFACTS.clear()
        artifacts_defs.ARTIFACTS.extend(original_artifacts)


def test_initialize_agent_compiles_with_sparse_artifact_ids(monkeypatch):
    original_artifacts = list(artifacts_defs.ARTIFACTS)

    def passthrough_choice_agent(**_kwargs):
        artifacts_defs.ARTIFACTS.clear()
        artifacts_defs.ARTIFACTS.extend(
            [
                _artifact(0, "First"),
                _artifact(1, "Second"),
            ]
        )
        return lambda state: state

    try:
        artifacts_defs.ARTIFACTS.clear()
        artifacts_defs.ARTIFACTS.extend(
            [
                _artifact(0, "First"),
                _artifact(1, "Second"),
                _artifact(5, "Third"),
                _artifact(12, "Fourth"),
            ]
        )
        monkeypatch.setattr(theodor_agent, "set_global_locale", lambda locale: locale)
        monkeypatch.setattr(theodor_agent, "build_choice_agent", passthrough_choice_agent)
        monkeypatch.setattr(theodor_agent, "JSONFileTracer", lambda *_args, **_kwargs: BaseCallbackHandler())
        monkeypatch.setattr(theodor_agent.config, "LANGFUSE_URL", "")

        graph = theodor_agent.initialize_agent(provider=ModelType.GPT, locale="ru")

        assert graph is not None
    finally:
        artifacts_defs.ARTIFACTS.clear()
        artifacts_defs.ARTIFACTS.extend(original_artifacts)
