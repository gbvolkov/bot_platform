from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .state.state import CommonAgentState
from user_manager.utils import UserManager

@tool
def fetch_user_info(config: RunnableConfig)  -> dict:
    """Fetch all information of the user's profile along.

    Returns:
        Dictionary containing the user profile details.
    """
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)
    user_role = configuration.get("user_role", "default")
    if not user_id:
        raise ValueError("No user ID configured.")
    return {"user_id": user_id, "user_role": user_role}

def user_info(state: CommonAgentState, config: RunnableConfig):
    # TODO: This should come from telegram chat bot info
    return {"user_info": fetch_user_info.invoke(config)}
