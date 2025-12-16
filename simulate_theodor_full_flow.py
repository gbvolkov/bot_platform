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
from agents.theodor_agent.artifacts_defs import ArtifactAgentContext, ArtifactState

from agents.llm_utils import (
    get_llm,
    with_llm_fallbacks,
)
_PRIMARY_RETRY_ATTEMPTS = 3


import time

_MAX_ATTTEMPTS = 3

USER_SIMULATOR_PROMPT = """
Ты - ответственный секретарь Управляюшего комитета по новым продуктам страховой компани Ингосстрах.
Ты отвечаешь за то, чтобы предоставляемые материалы были качественно проработаны и потенциально приносили ценность компании.
Ты оцениваешь идеи с точки зрения методологии RICE.

Твой собеседник - продуктолог. Он прорабатывает идею нового продукта для компании и предоставляет тебе различные варианты артефактов по методологии 13 шагов.
Твоя задача - обеспечить максимальное качество материалов и оценить потенциальную ценность идеи.
Ты рассматриваешь предоставленные артефакты, критикуешь их, выбираешь один их предоставленных вариантов.

Вместе с твоим обеседником вы должны пройти все 13 шагов и получить лучший результат.

Там, где это уместно давай оценку вариантов/артефактов/идей с точки зрения RICE.
"""

#USER_SIMULATOR_PROMPT = """
#Ты подтверждаешь артефакт. 
#Если тебя просят выбрать один из вариантов: всегда отвечай "Вариант А".
#Если тебя просят подтвердить финальный вариант артефакта - всегда отвечай "подтверждаю".
#"""

async def run_simulation():

    print("Initializing Theodor Agent...")
    agent_graph = initialize_agent()
    _user_primary_llm = get_llm(model="base", provider="openai")
    _user_alternative_llm = get_llm(model="mini", provider="openai_4", temperature=0)
    user_llm = with_llm_fallbacks(
        _user_primary_llm,
        alternative_llm=_user_alternative_llm,
        primary_retries=_PRIMARY_RETRY_ATTEMPTS,
    )

    logging.info("Simulation started")

    config = {"configurable": {"thread_id": "simulation_thread_1"}}

    idea = (
        "###Идея:\n"
        "***Для покупателей квартир на вторичном рынке***,  \n"
        "которые боятся стать жертвами мошенников и потерять деньги и недвижимость,  \n"
        "**наш продукт** — это комплексный онлайн-сервис юридической и технической проверки квартиры,  \n"
        "**который** выявляет все скрытые риски и юридические проблемы до сделки,  \n"
        "**в отличие** от разрозненных проверок через Росреестр, нотариусов и риелторов,  \n"
        "**наш продукт** проводит автоматизированную проверку по 50+ параметрам за 24 часа с юридической гарантией результата."
    )

    next_input: Any = {
        "messages": [
            HumanMessage(
                content=idea
            )
        ]
    }

    max_steps = 50  # Safety limit
    #print("\n--- STARTING SIMULATION ---\n")
    logging.info("\n--- STARTING SIMULATION ---\n")

    ctx = ArtifactAgentContext(user_prompt = idea,
                               generated_artifacts = [])


    from agents.utils import show_graph

    #show_graph(agent_graph)
    current_artifact_id = -1
    current_state = ArtifactState.INIT
    attempts = 0

    SIMULATE = True
    with open("docs/simulate_theodor_full_flow.md", "w", encoding="utf-8") as f:
        while True:
            events = agent_graph.invoke(
                next_input,
                config=config,
                context=ctx,
                stream_mode=["messages", "updates"],
            )
            await asyncio.sleep(1)
            #res = agent_graph.get_state(config, subgraphs=True)
            state_snapshot = agent_graph.get_state(config)

            logging.info("Current step index: %s", current_artifact_id)
            print("=========================================================================\n\n")
            f.write("=========================================================================\n\n")

            interrupt_payload = None
            last_ai_message: str = ""
            last_ai_node: str = ""

            for mode, payload in events or []:
                if mode == "messages":
                    msg, meta = payload
                    if not isinstance(msg, AIMessage):
                        continue
                    node_name = ""
                    if isinstance(meta, dict):
                        node_name = str(meta.get("langgraph_node") or "")

                    raw_content = msg.content
                    msg_text = raw_content.strip() if isinstance(raw_content, str) else str(raw_content or "").strip()
                    if msg_text:
                        last_ai_message = msg_text
                        last_ai_node = node_name

                    if not msg_text or node_name == "init" or node_name.startswith(("choice_agent_", "cleanup_")):
                        continue

                    message = f"\n**Ответ продуктолога**:\n{msg_text}\n\n"
                    print(message)
                    f.write(message)
                elif mode == "updates" and isinstance(payload, dict) and "__interrupt__" in payload:
                    interrupts = payload.get("__interrupt__") or ()
                    latest = interrupts[-1] if interrupts else None
                    interrupt_payload = getattr(latest, "value", latest)

            if isinstance(interrupt_payload, dict):
                last_ai = interrupt_payload.get("content", "")

                current_artifact_id_new = state_snapshot.values.get("current_artifact_id", 0)
                current_state_new = interrupt_payload.get("current_artifact_state", ArtifactState.INIT)
                if current_artifact_id_new != current_artifact_id or current_state != current_state_new:
                    attempts = 1
                    current_artifact_id = current_artifact_id_new
                    current_state = current_state_new
                else:
                    attempts = attempts + 1

                #print(f"Interrupt payload: {payload}")
                if SIMULATE:
                    message = f"\n**Ответ продуктолога**:\n{last_ai}\n" 
                    print(message)
                    f.write(message)
                    prompt = f"{message}\nТвой ответ (это {attempts} итерация):"
                    
                    sys_prompt = USER_SIMULATOR_PROMPT
                    if attempts >= _MAX_ATTTEMPTS:
                        if current_state in (ArtifactState.INIT, ArtifactState.OPTIONS_GENERATED):
                            sys_prompt += "\n**ВАЖНО**: Сейчас ты ДОЛЖЕН выбрать один из предложенных тебе вариантов. Дальнейшие изменения НЕВОЗМОЖНЫ! Не давай замечаний - просто выбери вариант."
                        else:
                            sys_prompt += "\n**ВАЖНО**: Сейчас ты ДОЛЖЕН подтвердить арефакт словом 'подтверждаю'.\nДальнейшие изменения НЕВОЗМОЖНЫ! Не давай замечаний - просто скажи 'подтверждаю'."

                    sim_messages = [
                        SystemMessage(content=sys_prompt),
                        HumanMessage(content=prompt),
                    ]

                    user_reply: str = user_llm.invoke(sim_messages).content
                    print("-----------------------------------------------------------------\n")
                    f.write("-----------------------------------------------------------------\n")
                    message = f"\n**Ответ эксперта**:\n {user_reply}"
                    print(message)
                    f.write(message)
                else:
                    message = f"\n**Ответ продуктолога**:\n{last_ai}\n\n"
                    print(message)
                    f.write(message)
                    user_reply = input("\n**Ответ эксперта**:")
                next_input = Command(resume=user_reply)
                await asyncio.sleep(1)
                continue

            if last_ai_message and last_ai_node.startswith("choice_agent_"):
                message = f"\n**Ответ продуктолога**:\n{last_ai_message}\n\n"
                print(message)
                f.write(message)

            if len(state_snapshot.next) == 0:
                artifacts = state_snapshot.values.get("artifacts") or {}
                out_path = Path("docs") / "simulate_theodor_artifacts.md"
                out_path.parent.mkdir(exist_ok=True)

                if isinstance(artifacts, dict):
                    artifact_items = ((k, artifacts[k]) for k in sorted(artifacts))
                else:
                    artifact_items = enumerate(artifacts)

                parts: List[str] = []
                append = parts.append
                for artifact_id, details in artifact_items:
                    details = details or {}
                    definition = details.get("artifact_definition") or {}
                    name = definition.get("name") or f"Artifact {artifact_id + 1}"
                    append(f"## {artifact_id + 1}. {name}\n")
                    append((details.get("artifact_final_text") or "").strip())
                    append("")

                final_text = "\n".join(parts).rstrip() + "\n"
                out_path.write_text(final_text, encoding="utf-8")
                print(final_text)
                break
            #break

    
    print("ФСЁ")


if __name__ == "__main__":
    asyncio.run(run_simulation())
