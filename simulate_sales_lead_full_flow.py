from pathlib import Path
import asyncio
import logging
import sys
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agents.sales_lead_agent import initialize_agent
from agents.utils import extract_text

ATTACHMENT_PATHS: list[Path] = []
SIMULATION_TURNS: list[str] = [
    "Найди потенциальных лидов по закупкам и открытым источникам по теме страхования перевозок в ЦФО за последнюю неделю.",
    "Покажи short-list и поясни, какие лиды стоит брать в работу первыми.",
    "Сформируй выгрузку по найденным лидам и добавь короткое summary для продавца.",
]

LOG_DIR = Path(".")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "sales_lead_simulation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("sales_lead_simulator")
dialog_handler = logging.FileHandler(LOG_DIR / "sales_lead_dialog.log", encoding="utf-8")
dialog_handler.setLevel(logging.INFO)
dialog_handler.setFormatter(logging.Formatter("%(asctime)s DIALOG %(message)s"))
logger.addHandler(dialog_handler)
sys.stdout.reconfigure(encoding="utf-8")


def _build_attachment_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for raw_path in paths:
        file_path = raw_path.expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Attachment not found: {file_path}")
        payloads.append(
            {
                "filename": file_path.name,
                "content_type": "application/octet-stream",
                "path": str(file_path),
            }
        )
    return payloads


def _latest_ai_message(result: dict[str, Any]) -> AIMessage | None:
    for message in reversed(result.get("messages") or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _message_attachments(message: BaseMessage | None) -> list[str]:
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []

    filenames: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "file":
            continue
        filename = str(item.get("filename") or "").strip()
        if filename:
            filenames.append(filename)
    return filenames


async def run_simulation() -> None:
    print("Initializing Sales Lead Agent...")
    agent_graph = initialize_agent(locale="ru")
    logging.info("Sales lead simulation started")

    config = {
        "configurable": {
            "thread_id": "sales_lead_simulation_thread",
            "user_id": "sales_lead_simulator",
            "user_role": "user",
        }
    }
    attachments = _build_attachment_payloads(ATTACHMENT_PATHS)

    for index, user_text in enumerate(SIMULATION_TURNS, start=1):
        print(f"\nUser ({index}):\n{user_text}\n")
        logger.info("User: %s", user_text)

        turn_state: dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}
        if index == 1 and attachments:
            turn_state["attachments"] = attachments

        result = await agent_graph.ainvoke(turn_state, config=config)
        ai_message = _latest_ai_message(result)
        if ai_message is None:
            print("No AI message, stopping.")
            logger.warning("No AI message returned on turn %s", index)
            break

        ai_text = extract_text(ai_message)
        print(f"AI:\n{ai_text}\n")
        logger.info("AI: %s", ai_text)

        exported_files = _message_attachments(ai_message)
        if exported_files:
            print("Attachments:")
            for filename in exported_files:
                print(f"- {filename}")
                logger.info("Attachment: %s", filename)
            print("")

        await asyncio.sleep(0.3)

    print("Simulation finished.")


if __name__ == "__main__":
    asyncio.run(run_simulation())
