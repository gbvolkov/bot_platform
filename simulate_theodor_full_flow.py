from pathlib import Path
import asyncio
import logging
# Configure logging (console + UTF-8 file)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "simulation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("simulator")

from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from agents.theodor_agent.agent import initialize_agent
from agents.theodor_agent.artifacts_defs import ArtifactAgentContext
from agents.utils import get_llm


import time

USER_SIMULATOR_PROMPT = """
Ты - ответственный секретарь Управляюшего комитета по новым продуктам страховой компани Ингосстрах.
Ты отвечаешь за то, чтобы предоставляемые материалы были качественно проработаны и потенциально приносили ценность компании.
Ты оцениваешь идеи с точки зрения методологии RICE.

Твой собеседник - продуктолог. Он прорабатывает идею нового продукта для компании и предоставляет тебе различные варианты артефактов по методологии 13 шагов.
Твоя задача - обеспечить максимальное качество материалов и оценить потенциальную ценность идеи.
Ты рассматриваешь предоставленные артефакты, критикуешь их, выбираешь один их предоставленных вариантов.

Вместе с твоим обеседником вы должны пройти все 13 шагов и получить лучший результат.

На каждый артефакт или расммотрение вариантов артефакта ты можешь тратить не более трёх итераций, после чео ты должен сделать выбор, зафиксировать окончательный вариант артефакта и двигаться дальше.
Там, где это уместно давай оценку вариантов/артефактов/идей с точки зрения RICE.
"""


async def run_simulation():

    print("Initializing Theodor Agent...")
    agent_graph = initialize_agent()
    user_llm = get_llm(model="base", provider="openai")
    logging.info("Simulation started")

    config = {"configurable": {"thread_id": "simulation_thread_1"}}

    next_input: Any = {
        "messages": [
            HumanMessage(
                content=(
                    "###Идея:\n"
                    "***Для покупателей квартир на вторичном рынке***,  \n"
                    "которые боятся стать жертвами мошенников и потерять деньги и недвижимость,  \n"
                    "**наш продукт** — это комплексный онлайн-сервис юридической и технической проверки квартиры,  \n"
                    "**который** выявляет все скрытые риски и юридические проблемы до сделки,  \n"
                    "**в отличие** от разрозненных проверок через Росреестр, нотариусов и риелторов,  \n"
                    "**наш продукт** проводит автоматизированную проверку по 50+ параметрам за 24 часа с юридической гарантией результата."
                )
            )
        ]
    }

    max_steps = 50  # Safety limit
    #print("\n--- STARTING SIMULATION ---\n")
    logging.info("\n--- STARTING SIMULATION ---\n")

    ctx = ArtifactAgentContext(user_prompt = "Привет! У меня есть идея стартапа: Uber для выгула собак. Помоги мне проработать её.",
                               generated_artifacts = [])

    SIMULATE = True
    with open("simulate_theodor_full_flow.md", "w", encoding="utf-8") as f:
        while True:
            result = agent_graph.invoke(next_input, config=config, context=ctx)
            await asyncio.sleep(1)
            #res = agent_graph.get_state(config, subgraphs=True)
            state_snapshot = agent_graph.get_state(config)
            if len(state_snapshot.next) == 0:
                break
            current_idx = state_snapshot.values.get("current_step_index", 0)

            logging.info("Current step index: %s", current_idx)
            interrupts = result.get("__interrupt__")
            if interrupts:
                payload = getattr(interrupts[-1], "value", interrupts[-1])
                last_ai = payload.get("content", "")
                #print(f"Interrupt payload: {payload}")
                if SIMULATE:
                    message = f"**Ответ продуктолога**:\n{last_ai}\n" 
                    print(message)
                    f.write(message)
                    prompt = f"{message}\nТвой ответ:"
                    sim_messages = [
                        SystemMessage(content=USER_SIMULATOR_PROMPT),
                        HumanMessage(content=prompt),
                    ]

                    user_reply: str = user_llm.invoke(sim_messages).content
                    print("-----------------------------------------------------------------\n")
                    f.write("-----------------------------------------------------------------\n")
                    message = f"**Ответ эксперта**:\n {user_reply}"
                    print(message)
                    f.write(message)
                else:
                    message = f"**Ответ продуктолога**:\n{last_ai}\n\n"
                    print(message)
                    f.write(message)
                    user_reply = input("**Ответ эксперта**:")
                next_input = Command(resume=user_reply)
                await asyncio.sleep(1)
                continue
            else:
                last_ai = ""
                if result.get("messages"):
                    for msg in reversed(result["messages"]):
                        if isinstance(msg, AIMessage):
                            last_ai = msg.content
                            break
                if last_ai:
                    #print(f"Bot: {last_ai}")
                    message = f"**Ответ продуктолога**:\n{last_ai}\n\n"
                    print(message)
                    f.write(message)
                    #logging.info("Bot: %s", last_ai)
            #break
            print("=========================================================================\n\n")
            f.write("=========================================================================\n\n")

    print("ФСЁ")


if __name__ == "__main__":
    asyncio.run(run_simulation())
