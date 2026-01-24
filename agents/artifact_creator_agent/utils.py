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

    is_change_requested: Annotated[
        bool,
        Field(description="Did user request changed. True or False."),
    ]

    # NotRequired[str] -> optional field in Pydantic: give it a default
    change_request: Annotated[
        str | None,
        Field(default=None, description="Request to change."),
    ] = None

    @model_validator(mode="after")
    def validate_change_request(self) -> "UserChangeRequest":
        # Cross-field validation (v2 replacement for most root_validator use cases)
        if self.is_change_requested and not (self.change_request and self.change_request.strip()):
            raise ValueError("change_request must be provided when is_change_requested=True")
        return self


def _is_user_confirmed(text: str, artifact_text: str):
    """
    Lightweight LLM-based classifier to judge approval/confirmation.
    Falls back to keyword match if the LLM call fails.
    """
    #normalized = text.lower().strip().strip(".,!?;")
    clf = _response_analyser_llm.with_structured_output(UserChangeRequest)
    system = SystemMessage(
        content=(
            "User provided response to generated artifact:\n" 
            f"{artifact_text}\n"
            "You have to analise user's response.\n" 
            "Determine if user confirmed the text or requested change."
        )
    )
    user = HumanMessage(content=f"User reply: {text}")
    
    result = clf.invoke([system, user])
    if isinstance(result, dict):
        return result
    elif isinstance(result, UserChangeRequest):
        return result.model_dump()
