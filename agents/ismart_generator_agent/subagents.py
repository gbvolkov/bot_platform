from __future__ import annotations

from typing import Annotated, Any, Mapping, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from agents.structured_prompt_utils import provider_then_tool

from .schemas import (
    CurrentControlAutocheckSet,
    GeneratedMaterial,
    IntermediateAssessmentArtifact,
    MaterialValidationDecision,
    PackageValidationDecision,
    PracticeTaskInstanceSet,
    PracticeTaskTemplateSet,
    SelfWorkAutocheckSet,
    ValidationControllerDecision,
)


CONTENT_AGENT_TYPES: tuple[str, ...] = (
    "TheoryMaterialAgent",
    "PracticeMaterialAgent",
    "TeacherGuidanceAgent",
    "SelfStudyAgent",
    "CurrentControlAgent",
    "IntermediateAssessmentAgent",
    "FinalProjectAssessmentAgent",
    "SpecificationQAAgent",
)

VALIDATION_AGENT_TYPES: tuple[str, ...] = (
    "MaterialValidatorAgent",
    "ValidationControllerAgent",
    "PackageValidatorAgent",
)

PRACTICE_PIPELINE_AGENT_TYPES: tuple[str, ...] = (
    "PracticeTaskTemplateAgent",
    "PracticeTaskVariantAgent",
)

SELF_WORK_PIPELINE_AGENT_TYPES: tuple[str, ...] = (
    "SelfWorkAutocheckAgent",
)

CURRENT_CONTROL_PIPELINE_AGENT_TYPES: tuple[str, ...] = (
    "CurrentControlAutocheckAgent",
)

INTERMEDIATE_PIPELINE_AGENT_TYPES: tuple[str, ...] = (
    "IntermediateAssessmentArtifactAgent",
)

ALL_SUBAGENT_TYPES: tuple[str, ...] = (
    *CONTENT_AGENT_TYPES,
    *PRACTICE_PIPELINE_AGENT_TYPES,
    *SELF_WORK_PIPELINE_AGENT_TYPES,
    *CURRENT_CONTROL_PIPELINE_AGENT_TYPES,
    *INTERMEDIATE_PIPELINE_AGENT_TYPES,
    *VALIDATION_AGENT_TYPES,
)


class StructuredSubagentState(AgentState[dict[str, Any]]):
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: NotRequired[str]
    prompt: NotRequired[str]


def _build_structured_subagent(
    *,
    name: str,
    model: BaseChatModel,
    schema: type[BaseModel],
):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        return str((request.state or {}).get("system_prompt") or "")

    return create_agent(
        model=model,
        tools=None,
        middleware=[build_prompt, provider_then_tool],
        response_format=schema,
        state_schema=StructuredSubagentState,
    )


def build_subagent_registry(model: BaseChatModel) -> Mapping[str, Any]:
    """Build all explicit iSMART subagents as compiled LangGraph agents."""

    registry: dict[str, Any] = {}
    for agent_type in CONTENT_AGENT_TYPES:
        registry[agent_type] = _build_structured_subagent(
            name=agent_type,
            model=model,
            schema=GeneratedMaterial,
        )

    registry["PracticeTaskTemplateAgent"] = _build_structured_subagent(
        name="PracticeTaskTemplateAgent",
        model=model,
        schema=PracticeTaskTemplateSet,
    )
    registry["PracticeTaskVariantAgent"] = _build_structured_subagent(
        name="PracticeTaskVariantAgent",
        model=model,
        schema=PracticeTaskInstanceSet,
    )
    registry["SelfWorkAutocheckAgent"] = _build_structured_subagent(
        name="SelfWorkAutocheckAgent",
        model=model,
        schema=SelfWorkAutocheckSet,
    )
    registry["CurrentControlAutocheckAgent"] = _build_structured_subagent(
        name="CurrentControlAutocheckAgent",
        model=model,
        schema=CurrentControlAutocheckSet,
    )
    registry["IntermediateAssessmentArtifactAgent"] = _build_structured_subagent(
        name="IntermediateAssessmentArtifactAgent",
        model=model,
        schema=IntermediateAssessmentArtifact,
    )

    registry["MaterialValidatorAgent"] = _build_structured_subagent(
        name="MaterialValidatorAgent",
        model=model,
        schema=MaterialValidationDecision,
    )
    registry["ValidationControllerAgent"] = _build_structured_subagent(
        name="ValidationControllerAgent",
        model=model,
        schema=ValidationControllerDecision,
    )
    registry["PackageValidatorAgent"] = _build_structured_subagent(
        name="PackageValidatorAgent",
        model=model,
        schema=PackageValidationDecision,
    )
    return registry
