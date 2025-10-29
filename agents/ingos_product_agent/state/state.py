from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages, Messages
from langgraph.managed import IsLastStep, RemainingSteps

from ...utils import ModelType
from ...state.state import CommonAgentState, add_messages_no_img

class ProductAgentState(CommonAgentState):
    product: str
    verification_result: str
    verification_reason: str
