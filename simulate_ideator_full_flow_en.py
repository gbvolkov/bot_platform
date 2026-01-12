from pathlib import Path
import asyncio
import logging
import sys

from langchain_core.messages import AIMessage, HumanMessage

from agents.ideator_agent import initialize_agent
from agents.utils import get_llm

REPORT_PATH = Path("docs/ideator/report.json")

# Configure logging (console + UTF-8 file)
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
    agent_graph = initialize_agent(locale="en")
    user_llm = get_llm(model="mini", provider="openai")
    logging.info("Simulation started")

    config = {
        "configurable": {
            "thread_id": "ideator_simulation_thread",
        }
    }
    # Simulate raw attachment pass-through (as provided by bot_service)
    initial_state = {
        "messages": [],
        "attachments": [
            {
                "filename": REPORT_PATH.name,
                "content_type": "application/json",
                "path": str(REPORT_PATH.resolve()),
            }
        ],
    }

    # Initial user message to send into the graph
    messages = [HumanMessage(content="Hi! Please analyze the report and generate product ideas.")]
    dialog_history = list(messages)

    max_steps = 12
    for step in range(max_steps):
        result = agent_graph.invoke(initial_state | {"messages": messages}, config=config)
        state_snapshot = agent_graph.get_state(config)
        phase = state_snapshot.values.get("phase")

        # Grab the latest AI message
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

        # Simple auto-reply policy: prefer "1" unless we need to finish or clarify.
        dialog_history.append(ai_msg)
        base_prompt = (
            "You are simulating a user in a product ideation chat.\n"
            "Respond in English and keep replies short (one sentence or a single number).\n"
            "If the assistant offers a numbered list, reply with a single number (prefer 1).\n"
            "If the assistant asks an open question, answer with one short sentence.\n"
            "If the assistant asks whether to compare, rank, or refine, reply with \"compare\".\n"
            "If the assistant asks for your own option, provide a brief custom option in one sentence.\n"
            "If the assistant asks to continue or generate more ideas, reply with \"more ideas\".\n"
            "If the assistant asks to finish or send the idea, reply with \"finish\".\n"
        )
        history_text = "\n".join(
            f"{'Assistant' if isinstance(m, AIMessage) else 'User'}: {m.content}"
            for m in dialog_history
            if isinstance(m, (AIMessage, HumanMessage))
            and (getattr(m, "tool_calls", None) is None or getattr(m, "tool_calls") == [])
            and m.content
        )
        prompt = (
            base_prompt
            + "\n\nConversation history (latest last):\n"
            + history_text
        )
        user_reply = user_llm.invoke([AIMessage(content=prompt)]).content.strip()
        if not user_reply:
            user_reply = "1"

        print(f"User reply: {user_reply}")
        logger.info("User: %s", user_reply)
        if "finish" in user_reply.lower():
            print("Simulation finished.")
            break
        dialog_history.append(HumanMessage(content=user_reply))
        messages = [HumanMessage(content=user_reply)]
        await asyncio.sleep(0.5)

    print("Simulation finished.")


if __name__ == "__main__":
    asyncio.run(run_simulation())
