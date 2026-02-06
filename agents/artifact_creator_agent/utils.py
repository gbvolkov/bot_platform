from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents.llm_utils import (
    build_model_fallback_middleware,
    get_llm,
    with_llm_fallbacks,
)

_response_analyser_llm = get_llm(model="nano", provider="openai", temperature=0)

class UserConfirmation(BaseModel):
    """User change request.
    Запрос пользователя на изменение.
    """
    model_config = ConfigDict(extra="forbid")

    is_artifact_confirmed: Annotated[
        bool,
        Field(description="Did user finally confirmed the artifact. True or False."),
    ]

    