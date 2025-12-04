from pathlib import Path
import asyncio
import logging
import sys

from langchain_core.messages import AIMessage, HumanMessage

from agents.ideator_agent import initialize_agent
from agents.utils import get_llm

REPORT_PATH = Path("docs/ideator/batch_report_2025_11_28_10_00_51_490647.json")

# Configure logging (console + UTF-8 file)
#LOG_DIR = Path("logs")
LOG_DIR = Path(".")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "simulation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ideator_simulator")
dialog_handler = logging.FileHandler(LOG_DIR / "ideator.log", encoding="utf-8")
dialog_handler.setLevel(logging.INFO)
dialog_handler.setFormatter(logging.Formatter("%(asctime)s DIALOG %(message)s"))
logger.addHandler(dialog_handler)
sys.stdout.reconfigure(encoding="utf-8")


async def run_simulation():
    print("Initializing Ideator Agent...")
    agent_graph = initialize_agent()
    user_llm = get_llm(model="mini", provider="openai")
    logging.info("Simulation started")

    config = {"configurable": {"thread_id": "ideator_simulation_thread"}}
    ctx = {"report_path": str(REPORT_PATH)}

    # last user message to send into the graph
    messages = [HumanMessage(content="Привет! Запусти генератор идей по отчёту Разведчика.")]
    dialog_history = list(messages)

    max_steps = 12
    for step in range(max_steps):
        result = agent_graph.invoke({"messages": messages}, config=config, context=ctx)
        state_snapshot = agent_graph.get_state(config)
        phase = state_snapshot.values.get("phase")

        # grab latest AI message
        ai_msg = None
        if result.get("messages"):
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    ai_msg = msg
                    break
        if ai_msg:
            print(f"\nAI:\n{ai_msg.content}\n")
            logger.info("AI: %s", ai_msg.content)
        else:
            print("No AI message, stopping.")
            break

        if phase == "finish":
            print("Phase finish reached. Stopping simulation.")
            break

        # simple auto-reply policy: always reply "1" unless we need "ещё"
        dialog_history.append(ai_msg)
        base_prompt = (
            "Ты отвечаешь за развитие бизнеса в крупной страховой компании.\n"
            "Твоя задача - сгенерировать идею для последующего воплощения в твоей компании.\n"
            "Твой собеседник -  консультант-ментор по генерации идей. Ты консультируешься ним. Он помогает тебе сгенерировать идею для твоего бизнеса.\n"
            "Отвечай на его вопросы и предлагай изменения.\n"
            "Ты являешься инициирующей стороной и заказчиком, так что инициатива - за тобой.\n"
            "Ментор консультирует тебя, а не ты - его.\n"
            "Ты критикуешь его совет и предлагаешь к рассмотрению свои идеи.\n"
            "Ты не спрашиваешь ментора, что он хочет от тебя - это ты хочешь от него!!!! Ты - заказчик!!!!!\n"
            "На обсуждение смысловой линии или идеи тебе отводится не более 4 раз. После этого ты должен выбрать один из вариантов и подтвердить дальнейшие шаги.\n"
            "Не бери на себя роль ведущего! Ты не ведёшь беседу. Ведёт беседу - ментор! Ты критикуешь то, что он говорит или принимаешь его предложения, или предлагаешь свои!\n"
            "Если ты полностью удовлетворён результатом беседы с ментором - верни ровно одно слово: 'Довольно.'.\n"
        )
        history_text = "\n".join(
            f"{'Консультант-ментор' if isinstance(m, AIMessage) else 'Ты (заказчик)'}: {m.content}"
            for m in dialog_history
            if isinstance(m, (AIMessage, HumanMessage))
            and (getattr(m, "tool_calls", None) is None or getattr(m, "tool_calls") == [])
            and m.content
        )
        prompt = (
            base_prompt
            + "\n\nКонтекст диалога (используй его, но оставайся в роли заказчика):\n"
            + history_text
        )
        user_reply = user_llm.invoke([AIMessage(content=prompt)]).content.strip()
        if not user_reply:
            user_reply = "1"

        print(f"User reply: {user_reply}")
        logger.info("User: %s", user_reply)
        if "довольно" in user_reply.lower():
            print("Simulation finished.")
            break
        dialog_history.append(HumanMessage(content=user_reply))
        messages = [HumanMessage(content=user_reply)]
        await asyncio.sleep(0.5)

    print("Simulation finished.")


if __name__ == "__main__":
    asyncio.run(run_simulation())
