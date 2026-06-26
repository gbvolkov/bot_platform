from __future__ import annotations

from typing import Any, Mapping, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .schemas import (
    GeneratedMaterial,
    MaterialValidationDecision,
    PackageValidationDecision,
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

ALL_SUBAGENT_TYPES: tuple[str, ...] = (*CONTENT_AGENT_TYPES, *VALIDATION_AGENT_TYPES)


class StructuredSubagentState(TypedDict, total=False):
    system_prompt: str
    prompt: str
    result: Any


def _build_structured_subagent(
    *,
    name: str,
    model: BaseChatModel,
    schema: type[BaseModel],
):
    structured_model = model.with_structured_output(schema)

    def invoke_model(state: StructuredSubagentState) -> dict[str, Any]:
        messages = []
        system_prompt = state.get("system_prompt") or ""
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=state.get("prompt") or ""))
        return {"result": structured_model.invoke(messages)}

    builder = StateGraph(StructuredSubagentState)
    builder.add_node("structured_output", invoke_model)
    builder.add_edge(START, "structured_output")
    builder.add_edge("structured_output", END)
    return builder.compile(name=name)


def build_subagent_registry(model: BaseChatModel) -> Mapping[str, Any]:
    """Build all explicit iSMART subagents as compiled LangGraph graphs."""

    registry: dict[str, Any] = {}
    for agent_type in CONTENT_AGENT_TYPES:
        registry[agent_type] = _build_structured_subagent(
            name=agent_type,
            model=model,
            schema=GeneratedMaterial,
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
