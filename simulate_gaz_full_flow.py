from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.gaz_agent.agent import initialize_agent
from agents.llm_utils import get_llm
from agents.utils import extract_text


@dataclass(frozen=True)
class GazPersona:
    persona_id: str
    name: str
    description: str
    opening_message: str
    behavior_prompt: str
    max_turns: int = 10


class PersonaReply(BaseModel):
    reply: str = Field(..., min_length=1)
    should_end: bool = False


PERSONAS: Dict[str, GazPersona] = {
    "cold_chain_owner": GazPersona(
        persona_id="cold_chain_owner",
        name="Владелец городской холодной доставки",
        description="Собственник доставки охлажденных продуктов. Сравнивает ГАЗ с Sollers Atlant, чувствителен к цене, быстро раздражается от лишних вопросов.",
        opening_message="Здравствуйте. У меня городская доставка охлажденки, смотрю на рефрижератор. Что у вас вообще есть и чем это лучше Atlant?",
        behavior_prompt=(
            "Ты владелец небольшой компании по городской доставке охлажденных продуктов. "
            "Тебе нужны практичные ответы: какие семейства подходят, что по рефрижератору, как сравнивать с Atlant и на что смотреть по деньгам. "
            "Ты нетерпелив, не любишь анкеты, быстро режешь длинные ответы, но если агент дает конкретику и внятный следующий шаг, готов двигаться дальше."
        ),
        max_turns=9,
    ),
    "moving_founder": GazPersona(
        persona_id="moving_founder",
        name="Основатель сервиса переездов",
        description="Основатель сервиса квартирных и офисных переездов. Пока не знает точную модель, хочет широкий обзор и понятное сужение между Соболь/Газель/среднетоннажниками.",
        opening_message="Добрый день. У нас сервис переездов по городу и области. Коротко расскажите, что у вас есть под такую задачу и с чего вообще смотреть?",
        behavior_prompt=(
            "Ты основатель сервиса квартирных и офисных переездов. Пока не хочешь глубоко погружаться в модификации, сначала хочешь получить широкий обзор, а потом уже сузить выбор. "
            "Ты дружелюбен, но если агент уходит в допрос, мягко возвращаешь его к сути: сначала что подходит, потом один уточняющий вопрос."
        ),
        max_turns=10,
    ),
    "municipal_procurement": GazPersona(
        persona_id="municipal_procurement",
        name="Муниципальный закупщик спецтехники",
        description="Специалист по закупке спецтехники для муниципалитета. Нужны надстройки, плохие условия эксплуатации, документы и аккуратный формальный тон.",
        opening_message="Здравствуйте. Нам нужна спецтехника для муниципальных задач, условия тяжелые, возможна надстройка. Какие направления у ГАЗ есть и что вы можете прислать на обоснование?",
        behavior_prompt=(
            "Ты специалист по закупкам муниципального предприятия. Тебе важны формальный тон, применимость к тяжелым условиям, спецнадстройки и пакет материалов для обоснования выбора. "
            "Ты терпелив и готов немного подождать ради документов, если агент честно объясняет, зачем это нужно."
        ),
        max_turns=10,
    ),
    "route_operator": GazPersona(
        persona_id="route_operator",
        name="Оператор пассажирского маршрута",
        description="Управляет пассажирским маршрутом. Смотрит на вместимость, формат салона, надежность и экономику маршрута.",
        opening_message="Добрый день. Мы обновляем подвижной состав на пассажирском маршруте. Какие у вас есть направления по автобусам и маршрутным решениям, и как между ними вообще выбирать?",
        behavior_prompt=(
            "Ты руководишь пассажирским маршрутом. Тебе важны вместимость, формат салона, надежность, обслуживание и экономика маршрута. "
            "Ты спокоен и аналитичен: любишь сравнения и структурированные ответы, но не хочешь читать каталог целиком."
        ),
        max_turns=10,
    ),
    "skeptical_cfo": GazPersona(
        persona_id="skeptical_cfo",
        name="Скептичный CFO по обновлению парка",
        description="Финансовый директор, который смотрит на TCO, лизинг, ежемесячную нагрузку и внутреннее согласование. Требует конкретики и может давить конкурентом.",
        opening_message="Здравствуйте. Я смотрю обновление парка доставки. Мне не нужен каталог, мне нужно понять, где у вас сильная экономика, что по финансированию и чем это лучше альтернативы на рынке.",
        behavior_prompt=(
            "Ты финансовый директор. Тебя интересуют TCO, финансовые программы, ежемесячная нагрузка, риски простоя и пакет на внутреннее согласование. "
            "Ты строгий, скептичный и быстро начинаешь давить конкурентом, если ответ слишком общий или без цифр и логики."
        ),
        max_turns=9,
    ),
}


def configure_stdio() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    elif hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    elif hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def configure_logging(persona_id: str) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"gaz_simulation_{persona_id}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger("gaz_simulator")
    logger.info("Writing simulation log to %s", log_path)
    return logger


def format_personas() -> str:
    lines: List[str] = []
    for index, persona in enumerate(PERSONAS.values(), start=1):
        lines.append(f"{index}. {persona.persona_id} — {persona.name}")
        lines.append(f"   {persona.description}")
    return "\n".join(lines)


def choose_persona(persona_id: str | None) -> GazPersona:
    if persona_id:
        if persona_id not in PERSONAS:
            raise ValueError(f"Unknown persona '{persona_id}'. Use --list-personas.")
        return PERSONAS[persona_id]

    print("Выберите персонажа:\n")
    print(format_personas())
    raw = input("\nВведите номер или persona_id: ").strip()
    if raw in PERSONAS:
        return PERSONAS[raw]
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError("Не удалось распознать выбор персонажа.") from exc
    personas = list(PERSONAS.values())
    if not (1 <= index <= len(personas)):
        raise ValueError("Номер персонажа вне диапазона.")
    return personas[index - 1]


def visible_text(message: BaseMessage) -> str:
    text = extract_text(message)
    return text.strip()


def latest_ai_text(result: dict) -> str:
    for message in reversed(result.get("messages") or []):
        if isinstance(message, AIMessage):
            return visible_text(message)
    return ""


def transcript_text(history: List[BaseMessage], limit: int = 14) -> str:
    visible: List[str] = []
    for message in history[-limit:]:
        text = visible_text(message)
        if not text:
            continue
        role = "Агент" if isinstance(message, AIMessage) else "Клиент"
        visible.append(f"{role}: {text}")
    return "\n".join(visible)


def build_customer_prompt(
    persona: GazPersona,
    history: List[BaseMessage],
    *,
    turn_index: int,
    max_turns: int,
    interrupt_payload: dict | None = None,
) -> List[BaseMessage]:
    transcript = transcript_text(history)
    if interrupt_payload:
        task_block = (
            "Сейчас агент не дал обычный ответ, а запросил подтверждение/ожидание через interrupt. "
            "Ответь именно на этот вопрос как клиент, не меняя тему.\n\n"
            f"Вопрос: {interrupt_payload.get('question') or ''}\n"
            f"Контекст interrupt: {interrupt_payload.get('content') or ''}"
        )
    else:
        latest = visible_text(history[-1]) if history else ""
        task_block = (
            "Ответь на последнюю реплику агента как живой клиент. "
            "Если агент полезен, двигай разговор к выбору, сравнению, материалам или следующему шагу. "
            "Если агент уходит в допрос или воду, реагируй в стиле персонажа.\n\n"
            f"Последняя реплика агента: {latest}"
        )

    system_prompt = (
        "Ты симулируешь клиента в B2B-диалоге с sales-агентом ГАЗ. "
        "Оставайся строго в роли клиента, не играй роль ассистента, не объясняй свои действия, не пиши ремарок. "
        "Отвечай на русском, 1-4 предложениями, без markdown и без списка. "
        "Если информации уже достаточно, чтобы принять следующий разумный шаг, ставь should_end=true и в reply дай естественную финальную реплику клиента.\n\n"
        f"Персонаж: {persona.name}.\n"
        f"Описание: {persona.description}\n"
        f"Поведение: {persona.behavior_prompt}\n"
        f"Текущий ход: {turn_index} из {max_turns}."
    )
    user_prompt = (
        f"История диалога:\n{transcript}\n\n"
        f"Задача на этот ход:\n{task_block}"
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def write_markdown(output_path: Path, persona: GazPersona, history: List[BaseMessage]) -> None:
    lines = [
        f"# Симуляция gaz_agent: {persona.name}",
        "",
        f"- Persona ID: `{persona.persona_id}`",
        f"- Описание: {persona.description}",
        "",
        "## Диалог",
        "",
    ]
    for message in history:
        text = visible_text(message)
        if not text:
            continue
        role = "Агент" if isinstance(message, AIMessage) else "Клиент"
        lines.append(f"### {role}")
        lines.append("")
        lines.append(text)
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_simulation(persona: GazPersona, turns_override: int | None, output_path: Path | None) -> Path:
    #configure_stdio()
    logger = configure_logging(persona.persona_id)
    max_turns = turns_override or persona.max_turns

    print(f"Initializing gaz_agent for persona: {persona.name}")
    agent_graph = initialize_agent(locale="ru")
    customer_llm = get_llm(model="base", provider="openai", temperature=0.7, streaming=False).with_structured_output(PersonaReply)

    thread_id = f"gaz_simulation_{persona.persona_id}_{time.strftime('%Y%m%d%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}

    history: List[BaseMessage] = [HumanMessage(content=persona.opening_message)]
    next_input: dict | Command = {"messages": [history[-1]]}
    stop_after_next_agent = False

    for turn_index in range(1, max_turns + 1):
        result = agent_graph.invoke(next_input, config=config)
        interrupts = result.get("__interrupt__") or []
        if interrupts:
            payload = getattr(interrupts[-1], "value", interrupts[-1]) or {}
            question = str(payload.get("question") or "").strip()
            content = str(payload.get("content") or question).strip()
            agent_text = question or content
            if agent_text:
                agent_message = AIMessage(content=agent_text)
                history.append(agent_message)
                print(f"\nАгент (interrupt):\n{agent_text}\n")
                logger.info("Agent interrupt: %s", agent_text)

            reply = customer_llm.invoke(
                build_customer_prompt(
                    persona,
                    history,
                    turn_index=turn_index,
                    max_turns=max_turns,
                    interrupt_payload=payload if isinstance(payload, dict) else {"content": str(payload)},
                )
            )
            customer_text = reply.reply.strip()
            history.append(HumanMessage(content=customer_text))
            print(f"Клиент:\n{customer_text}\n")
            logger.info("Customer resume: %s", customer_text)
            next_input = Command(resume=customer_text)
            continue

        agent_text = latest_ai_text(result)
        if not agent_text:
            logger.warning("No AI message returned on turn %s, stopping simulation.", turn_index)
            break

        history.append(AIMessage(content=agent_text))
        print(f"\nАгент:\n{agent_text}\n")
        logger.info("Agent: %s", agent_text)

        if stop_after_next_agent:
            logger.info("Stop flag reached after final agent answer.")
            break

        reply = customer_llm.invoke(
            build_customer_prompt(
                persona,
                history,
                turn_index=turn_index,
                max_turns=max_turns,
            )
        )
        customer_text = reply.reply.strip()
        history.append(HumanMessage(content=customer_text))
        print(f"Клиент:\n{customer_text}\n")
        logger.info("Customer: %s", customer_text)

        stop_after_next_agent = bool(reply.should_end)
        next_input = {"messages": [HumanMessage(content=customer_text)]}

    final_output = output_path or Path("docs") / "gaz" / "simulations" / f"simulate_gaz_{persona.persona_id}.md"
    write_markdown(final_output, persona, history)
    logger.info("Simulation transcript saved to %s", final_output)
    print(f"\nTranscript saved to: {final_output}")
    return final_output


def main() -> None:
    #configure_stdio()
    parser = argparse.ArgumentParser(description="Simulate a full sales dialogue with gaz_agent.")
    parser.add_argument("--persona", default="cold_chain_owner", help="Persona id to run. Use --list-personas to see options.")
    parser.add_argument("--all", action="store_true", default=True, help="Run all personas and exit. Use --list-personas to see options.")
    parser.add_argument("--list-personas", action="store_true", help="List available personas and exit.")
    parser.add_argument("--turns", type=int, default=5, help="Override max turns for the selected persona.")
    parser.add_argument("--output", type=Path, default=None, help="Path to save the transcript markdown.")
    args = parser.parse_args()

    if args.list_personas:
        print(format_personas())
        return

    if args.all:
        for persona in PERSONAS.values():
            run_simulation(persona, args.turns, args.output)
        return
    
    persona = choose_persona(args.persona)
    run_simulation(persona, args.turns, args.output)


if __name__ == "__main__":
    main()