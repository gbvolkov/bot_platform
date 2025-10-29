from typing import Annotated, Optional
from typing_extensions import TypedDict

from agents.ingos_product_agent.agent import initialize_agent
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
    query = "Допускается ли китайский автомобиль?"
    payload_msg = HumanMessage(content=[{"type": "text", "text": query}])
    agent = initialize_agent(product="Овертайм")  
    result = agent.invoke({"messages": [payload_msg]}, config=get_config(), stream_mode="values")

    print(result)


if __name__ == "__main__":
    main()
