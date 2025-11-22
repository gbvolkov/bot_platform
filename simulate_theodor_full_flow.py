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
from agents.utils import get_llm


USER_SIMULATOR_PROMPT = """
Ты — основатель стартапа "Uber для выгула собак".
Ты общаешься с "Продуктовым Наставником" (AI), который ведет тебя по методологии создания продукта.

Твоя задача:
1. Отвечать на вопросы наставника кратко и по делу.
2. Если наставник предлагает варианты (A, B, C), выбери один (например, "Выбираю вариант A").
3. Если наставник спрашивает "Подтверждаете?", отвечай "Подтверждаю".
4. Не задавай встречных вопросов, просто следуй процессу.
5. Твоя цель — пройти все 13 шагов как можно быстрее.
"""


async def run_simulation():
    print("Initializing Theodor Agent...")
    agent_graph = initialize_agent()
    user_llm = get_llm(model="base", provider="openai")
    logger.info("Simulation started")

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
    print("\n--- STARTING SIMULATION ---\n")
    logger.info("\n--- STARTING SIMULATION ---\n")

    for step_count in range(1, max_steps + 1):
        print(f"\n[ITERATION {step_count}]")
        logger.info(f"\n# [ITERATION {step_count}]")

        result = agent_graph.invoke(next_input, config=config)

        # Show last AI message (if any)
        last_ai = ""
        if result.get("messages"):
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    last_ai = msg.content
                    break
        if last_ai:
            print(f"Bot: {last_ai}")
            logger.info("Bot: %s", last_ai)

        # Completion check
        state_snapshot = agent_graph.get_state(config)
        current_idx = state_snapshot.values.get("current_step_index", 0)
        if current_idx >= 13:
            print("Simulation Finished! (All steps completed)")
            break

        # Handle interrupt payload (pause for user input)
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = getattr(interrupts[-1], "value", interrupts[-1])
            print(f"Interrupt payload: {payload}")
            #logger.info("Interrupt payload: %s", payload)

            sim_messages = [
                SystemMessage(content=USER_SIMULATOR_PROMPT),
                HumanMessage(content=f"Ответ наставника:\n{last_ai}\n\nТвой ответ:"),
            ]
            user_reply = user_llm.invoke(sim_messages).content
            print(f"User: {user_reply}")
            logger.info("User: %s", user_reply)

            next_input = Command(resume=user_reply)
            await asyncio.sleep(1)
            continue

        if not state_snapshot.next:
            print("Graph reached END without further interrupts. Stopping.")
            break

        print("No interrupt returned; exiting to avoid tight loop.")
        break


if __name__ == "__main__":
    asyncio.run(run_simulation())
