from __future__ import annotations

from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import AgentState
from langgraph.graph.message import AnyMessage, add_messages


class PersonProfile(TypedDict, total=False):
    name: str
    age: int
    school_year: int
    nosology_type: str


class PersonInfoExtraction(TypedDict, total=False):
    """Информация о профиле ученика для формирования подсказок."""

    name: Annotated[Optional[str], "Имя студента"]
    age: Annotated[Optional[int], "Возраст студента"]
    school_year: Annotated[Optional[int], "Год обучения"]
    nosology_type: Annotated[Optional[str], "Тип нозологии"]


class HintGenerationResponse(TypedDict, total=False):
    """Structured hint response."""

    hint: Annotated[Optional[str], "A single helpful hint (no full solution)"]


class IsmartTutorAgentContext(TypedDict, total=False):
    """Runtime context for the iSmart tutor agent."""

    person_id: str
    person_profile: PersonProfile


class IsmartTutorAgentState(AgentState[Dict[str, Any]]):
    """State schema for the iSmart tutor agent graph."""

    messages: Annotated[List[AnyMessage], add_messages]
    attachments: NotRequired[List[Dict[str, Any]]]
    user_info: NotRequired[Dict[str, Any]]
    last_user_text: NotRequired[str]
    pending_question: NotRequired[str]
    question: NotRequired[str]
    hint_raw: NotRequired[str]
    person_id: NotRequired[str]
    person_profile: NotRequired[PersonProfile]
    profile_complete_at_turn_start: NotRequired[bool]
    needs_person_info: NotRequired[bool]
    needs_question: NotRequired[bool]
    reset_done: NotRequired[bool]
