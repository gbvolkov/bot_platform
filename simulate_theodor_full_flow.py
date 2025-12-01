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


USER_SIMULATOR_PROMPT = """
Ты — основатель стартапа "Uber для выгула собак".
Ты общаешься с "Продуктовым Наставником" (AI), который ведет тебя по методологии создания продукта.

Твоя задача:
1. Отвечать на вопросы наставника кратко и по делу.
2. Если наставник предлагает варианты (A, B, C), выбери один (например, "Выбираю вариант A"),  либо предложи исправления.
3. Если наставник спрашивает "Подтверждаете?", отвечай "Подтверждаю", либо предложи исправления.
4. Твоя цель — пройти все 13 шагов и получить лучший результат.
"""


async def run_simulation():

    print("Initializing Theodor Agent...")
    agent_graph = initialize_agent()
    user_llm = get_llm(model="mini", provider="openai")
    logging.info("Simulation started")

    config = {"configurable": {"thread_id": "simulation_thread_1"}}

    next_input: Any = {
        "messages": [
            HumanMessage(
                content=(
                    "Привет! У меня есть идея стартапа: Uber для выгула собак. "
                    "Помоги мне проработать её."
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
    while True:
        result = agent_graph.invoke(next_input, config=config, context=ctx)
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
                prompt = f"Ответ наставника:\n{last_ai}\n\nТвой ответ:"
                print(prompt)
                sim_messages = [
                    SystemMessage(content=USER_SIMULATOR_PROMPT),
                    HumanMessage(content=prompt),
                ]

                user_reply: str = user_llm.invoke(sim_messages).content
                print(f"Your answer: {user_reply}")
            else:
                print(f"Ответ наставника:\n{last_ai}\n\n")
                user_reply = input("Твой ответ:")
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
                logging.info("Bot: %s", last_ai)
        #break

    print("ФСЁ")


if __name__ == "__main__":
    asyncio.run(run_simulation())
