from typing import Annotated, Optional
from typing_extensions import TypedDict

from agents.bi_agent.bi_agent import initialize_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig


from agents.state.state import ConfigSchema

def get_config():
    #configurable: ConfigSchema = {
    #    "user_id": 0,
    #    "user_role": "default",
    #    "thread_id": 0,
    #}

    #{"configurable": configurable}
    return RunnableConfig(ConfigSchema({"user_id": 100, "user_role": "default", "model": "openai", "thread_id": 110}))

def main():
    query = "Верни самые популярные страны."
    payload_msg = HumanMessage(content=[{"type": "text", "text": query}])
    agent = initialize_agent(notify_on_reload=False)  
    result = agent.invoke({"messages": [payload_msg]}, config=get_config(), stream_mode="values")


    print(result)


if __name__ == "__main__":
    main()
