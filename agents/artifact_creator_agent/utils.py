from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents.llm_utils import (
    build_model_fallback_middleware,
    get_llm,
    with_llm_fallbacks,
)

_response_analyser_llm = get_llm(model="nano", provider="openai", temperature=0)

class UserChangeRequest(BaseModel):
    """User change request.
    Запрос пользователя на изменение.
    """
    model_config = ConfigDict(extra="forbid")

    is_artifact_confirmed: Annotated[
        bool,
        Field(description="Did user finally confirmed the artifact. True or False."),
    ]

    # NotRequired[str] -> optional field in Pydantic: give it a default
    #change_request: Annotated[
    #    str | None,
    #    Field(default=None, description="Request to change."),
    #] = None

    #@model_validator(mode="after")
    #def validate_change_request(self) -> "UserChangeRequest":
    #    # Cross-field validation (v2 replacement for most root_validator use cases)
    #    if self.is_artifact_confirmed and not (self.change_request and self.change_request.strip()):
    #        raise ValueError("change_request must be provided when is_change_requested=True")
    #    return self


