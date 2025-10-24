from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages, Messages
from langgraph.managed import IsLastStep, RemainingSteps
from langchain.agents import AgentState

from ..utils import ModelType

def add_messages_no_img(msgs1: Messages, msgs2: Messages) -> Messages:
    # Need to clean up all user messages excepting the last message with type "human" (after which can follow messages with other types)
    msgs = [msg for msg in msgs1 if msg.type == "human"][::-1]
    if len(msgs) > 1:
        msg = msgs[1]
        cleaned_content = [content for content in msg.content if content.get("type", "") != "image_url"]
        msg.content = cleaned_content

    return add_messages(msgs1, msgs2)

class CommonAgentState(AgentState):
    messages: Annotated[list[AnyMessage], add_messages_no_img]
    user_info: str
    last_question: str
    agent_class: str

class ConfigSchema(TypedDict):
    user_id: Optional[str]
    user_role: Optional[str]
    model: Optional[str]
    thread_id: Optional[str]
