from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.llm_utils import get_llm
from agents.utils import ModelType, extract_text


DEFAULT_CONFIG_PATH = Path("agents/mycroft_agent/cli_config.json")
DEFAULT_OUTPUT_DIR = Path("docs/gaz/simulations")
DEFAULT_USER_ID = "mycroft-gaz-simulation"
DEFAULT_AGENT_LOCALE = {
    "save_confirmation": "[You can now download the file.]({url})",
}


@dataclass(frozen=True)
class GazPersona:
    persona_id: str
    name: str
    description: str
    opening_message: str
    behavior_prompt: str
    max_turns: int = 8


class PersonaReply(BaseModel):
    reply: str = Field(..., min_length=1)
    should_end: bool = False


PERSONAS: dict[str, GazPersona] = {
    "cold_chain_owner": GazPersona(
        persona_id="cold_chain_owner",
        name="Владелец городской холодной доставки",
        description=(
            "Собственник небольшой доставки охлажденных продуктов. Сравнивает ГАЗ "
            "с Sollers Atlant, чувствителен к цене и не любит длинные анкеты."
        ),
        opening_message=(
            "Здравствуйте. У меня городская доставка охлажденки, смотрю на "
            "рефрижератор. Что у ГАЗ вообще есть под такую задачу и чем это лучше Atlant?"
        ),
        behavior_prompt=(
            "Ты владелец небольшой компании по городской доставке охлажденных продуктов. "
            "Тебе нужны практичные ответы: какие семейства подходят, что с рефрижератором, "
            "как сравнивать с Atlant и на что смотреть по деньгам. Если агент уходит "
            "в длинный опрос, верни его к конкретике."
        ),
        max_turns=8,
    ),
    "moving_founder": GazPersona(
        persona_id="moving_founder",
        name="Основатель сервиса переездов",
        description=(
            "Основатель сервиса квартирных и офисных переездов. Хочет широкий обзор "
            "между Соболем, Газелью и среднетоннажными решениями."
        ),
        opening_message=(
            "Добрый день. У нас сервис переездов по городу и области. Коротко расскажите, "
            "что у ГАЗ есть под такую задачу и с чего вообще смотреть?"
        ),
        behavior_prompt=(
            "Ты основатель сервиса квартирных и офисных переездов. Сначала хочешь понять "
            "направления, потом сузить выбор. Если агент задает слишком много вопросов, "
            "попроси сначала дать 2-3 разумных варианта."
        ),
        max_turns=8,
    ),
    "municipal_procurement": GazPersona(
        persona_id="municipal_procurement",
        name="Муниципальный закупщик спецтехники",
        description=(
            "Специалист по закупке спецтехники для муниципальных задач. Нужны надстройки, "
            "эксплуатация в тяжелых условиях и материалы для обоснования выбора."
        ),
        opening_message=(
            "Здравствуйте. Нам нужна спецтехника для муниципальных задач, условия тяжелые, "
            "возможна надстройка. Какие направления у ГАЗ есть и что можно приложить "
            "к обоснованию?"
        ),
        behavior_prompt=(
            "Ты специалист по закупкам муниципального предприятия. Тебе важны применимость "
            "к тяжелым условиям, спецнадстройки, формальный тон и пакет материалов для "
            "обоснования выбора."
        ),
        max_turns=8,
    ),
    "route_operator": GazPersona(
        persona_id="route_operator",
        name="Оператор пассажирского маршрута",
        description=(
            "Управляет пассажирским маршрутом. Смотрит на вместимость, формат салона, "
            "надежность, обслуживание и экономику маршрута."
        ),
        opening_message=(
            "Добрый день. Мы обновляем подвижной состав на пассажирском маршруте. Какие "
            "у ГАЗ есть направления по автобусам и маршрутным решениям, и как между ними выбирать?"
        ),
        behavior_prompt=(
            "Ты руководишь пассажирским маршрутом. Тебе важны вместимость, формат салона, "
            "надежность, обслуживание и экономика маршрута. Любишь структурированные "
            "сравнения, но не хочешь читать каталог целиком."
        ),
        max_turns=8,
    ),
    "skeptical_cfo": GazPersona(
        persona_id="skeptical_cfo",
        name="Скептичный CFO по обновлению парка",
        description=(
            "Финансовый директор, который смотрит на TCO, лизинг, ежемесячную нагрузку "
            "и внутреннее согласование. Требует конкретики и может давить конкурентами."
        ),
        opening_message=(
            "Здравствуйте. Я смотрю обновление парка доставки. Мне не нужен каталог, мне "
            "нужно понять, где у ГАЗ сильная экономика, что по финансированию и чем это "
            "лучше альтернативы на рынке."
        ),
        behavior_prompt=(
            "Ты финансовый директор. Тебя интересуют TCO, финансовые программы, "
            "ежемесячная нагрузка, риски простоя и пакет на внутреннее согласование. "
            "Если ответ общий или без логики, требуй конкретики."
        ),
        max_turns=8,
    ),
}


def configure_stdio() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    elif hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    elif hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def configure_logging(persona_id: str) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"mycroft_gaz_simulation_{persona_id}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger("mycroft_gaz_simulator")
    logger.info("Writing simulation log to %s", log_path)
    return logger


def parse_provider(raw_value: str) -> ModelType:
    normalized = raw_value.strip().lower()
    for provider in ModelType:
        if normalized in {provider.value.lower(), provider.name.lower()}:
            return provider
    available = ", ".join(sorted(provider.value for provider in ModelType))
    raise ValueError(f"Unknown provider '{raw_value}'. Available values: {available}")


def format_personas() -> str:
    lines: list[str] = []
    for index, persona in enumerate(PERSONAS.values(), start=1):
        lines.append(f"{index}. {persona.persona_id} - {persona.name}")
        lines.append(f"   {persona.description}")
    return "\n".join(lines)


def choose_persona(persona_id: str) -> GazPersona:
    if persona_id not in PERSONAS:
        raise ValueError(f"Unknown persona '{persona_id}'. Use --list-personas.")
    return PERSONAS[persona_id]


def visible_text(message: BaseMessage) -> str:
    return extract_text(message).strip()


def latest_ai_text(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages") or []):
        if isinstance(message, AIMessage):
            return visible_text(message)
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
    return ""


def transcript_text(history: list[BaseMessage], limit: int = 14) -> str:
    visible: list[str] = []
    for message in history[-limit:]:
        text = visible_text(message)
        if not text:
            continue
        role = "Mycroft" if isinstance(message, AIMessage) else "Клиент"
        visible.append(f"{role}: {text}")
    return "\n".join(visible)


def interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return None
    payload = getattr(interrupts[-1], "value", interrupts[-1])
    if isinstance(payload, dict):
        return payload
    return {"content": str(payload)}


def interrupt_text(payload: dict[str, Any]) -> str:
    question = str(payload.get("question") or "").strip()
    content = str(payload.get("content") or "").strip()
    return question or content or "Mycroft requested additional input."


def build_customer_prompt(
    persona: GazPersona,
    history: list[BaseMessage],
    *,
    turn_index: int,
    max_turns: int,
    interrupt: dict[str, Any] | None = None,
) -> list[BaseMessage]:
    transcript = transcript_text(history)
    if interrupt is not None:
        task = (
            "Агент запросил дополнительный ввод через interrupt. Ответь как клиент "
            "на этот вопрос, не меняя тему.\n\n"
            f"Вопрос: {interrupt_text(interrupt)}"
        )
    else:
        latest = visible_text(history[-1]) if history else ""
        task = (
            "Ответь на последнюю реплику Mycroft как живой B2B-клиент. Если ответ "
            "полезен, двигай разговор к выбору, сравнению, цене, TCO, материалам или "
            "следующему шагу. Если ответ слишком общий или уходит в анкету, попроси "
            "конкретику в стиле персонажа.\n\n"
            f"Последняя реплика Mycroft: {latest}"
        )

    system_prompt = (
        "Ты симулируешь клиента в B2B-диалоге с sales-оркестратором ГАЗ. "
        "Оставайся строго в роли клиента. Не играй ассистента, не объясняй свои действия, "
        "не используй markdown. Отвечай по-русски, 1-4 предложениями. "
        "Если информации достаточно для разумного следующего шага, поставь should_end=true "
        "и дай естественную финальную реплику клиента.\n\n"
        f"Персонаж: {persona.name}\n"
        f"Описание: {persona.description}\n"
        f"Поведение: {persona.behavior_prompt}\n"
        f"Текущий ход: {turn_index} из {max_turns}."
    )
    user_prompt = f"История диалога:\n{transcript}\n\nЗадача на этот ход:\n{task}"
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def write_markdown(
    output_path: Path,
    *,
    persona: GazPersona,
    thread_id: str,
    config_path: Path,
    provider: ModelType,
    model_size: str,
    history: list[BaseMessage],
) -> None:
    lines = [
        f"# Симуляция Mycroft GAZ: {persona.name}",
        "",
        f"- Persona ID: `{persona.persona_id}`",
        f"- Thread ID: `{thread_id}`",
        f"- Config: `{config_path}`",
        f"- Provider: `{provider.value}`",
        f"- Model size: `{model_size}`",
        f"- Описание: {persona.description}",
        "",
        "## Диалог",
        "",
    ]
    for message in history:
        text = visible_text(message)
        if not text:
            continue
        role = "Mycroft" if isinstance(message, AIMessage) else "Клиент"
        lines.append(f"### {role}")
        lines.append("")
        lines.append(text)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def invoke_mycroft_turn(
    agent: Any,
    *,
    pending_input: dict[str, Any] | Command,
    config: dict[str, Any],
) -> dict[str, Any]:
    return agent.invoke(pending_input, config=config)


def run_simulation(
    persona: GazPersona,
    *,
    provider: ModelType,
    config_path: Path,
    model_size: str,
    customer_model_size: str,
    temperature: float,
    customer_temperature: float,
    turns_override: int | None,
    output_path: Path,
    user_id: str,
) -> Path:
    logger = configure_logging(persona.persona_id)
    max_turns = turns_override or persona.max_turns

    print(f"Initializing Mycroft GAZ agent for persona: {persona.name}")
    from agents.mycroft_agent.configured_agent import initialize_agent as initialize_mycroft_agent

    agent = initialize_mycroft_agent(
        provider=provider,
        config_path=config_path,
        model_size=model_size,
        temperature=temperature,
        streaming=False,
    )
    customer_llm = get_llm(
        model=customer_model_size,
        provider=provider.value,
        temperature=customer_temperature,
        streaming=False,
    ).with_structured_output(PersonaReply)

    thread_id = f"mycroft_gaz_simulation_{persona.persona_id}_{uuid.uuid4().hex}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }

    history: list[BaseMessage] = [HumanMessage(content=persona.opening_message)]
    pending_input: dict[str, Any] | Command = {
        "messages": [history[-1]],
        "locale": DEFAULT_AGENT_LOCALE,
    }
    stop_after_next_agent = False

    try:
        for turn_index in range(1, max_turns + 1):
            result = invoke_mycroft_turn(agent, pending_input=pending_input, config=config)
            interrupt = interrupt_payload(result)
            if interrupt is not None:
                if "action_requests" in interrupt:
                    raise RuntimeError(
                        "Simulation stopped because Mycroft requested a human approval action. "
                        f"Interrupt payload: {interrupt}"
                    )
                agent_text = interrupt_text(interrupt)
                history.append(AIMessage(content=agent_text))
                print(f"\nMycroft (interrupt):\n{agent_text}\n")
                logger.info("Mycroft interrupt: %s", agent_text)

                reply = customer_llm.invoke(
                    build_customer_prompt(
                        persona,
                        history,
                        turn_index=turn_index,
                        max_turns=max_turns,
                        interrupt=interrupt,
                    )
                )
                customer_text = reply.reply.strip()
                history.append(HumanMessage(content=customer_text))
                print(f"Клиент:\n{customer_text}\n")
                logger.info("Customer resume: %s", customer_text)
                pending_input = Command(resume=customer_text)
                continue

            agent_text = latest_ai_text(result)
            if not agent_text:
                logger.warning("No AI message returned on turn %s, stopping simulation.", turn_index)
                break

            history.append(AIMessage(content=agent_text))
            print(f"\nMycroft:\n{agent_text}\n")
            logger.info("Mycroft: %s", agent_text)

            if stop_after_next_agent:
                logger.info("Stop flag reached after final Mycroft answer.")
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
            pending_input = {
                "messages": [HumanMessage(content=customer_text)],
                "locale": DEFAULT_AGENT_LOCALE,
            }
    finally:
        from bot_service.agent_registry import agent_registry

        asyncio.run(agent_registry.aclose())

    write_markdown(
        output_path,
        persona=persona,
        thread_id=thread_id,
        config_path=config_path,
        provider=provider,
        model_size=model_size,
        history=history,
    )
    logger.info("Simulation transcript saved to %s", output_path)
    print(f"\nTranscript saved to: {output_path}")
    return output_path


def build_output_path(
    *,
    persona: GazPersona,
    output: Path | None,
    output_dir: Path,
    all_mode: bool,
) -> Path:
    if all_mode and output is not None:
        raise ValueError("--output cannot be used with --all. Use --output-dir.")
    if output is not None:
        return output
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return output_dir / f"simulate_mycroft_gaz_{persona.persona_id}_{timestamp}.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a full GAZ sales dialogue through Mycroft."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--provider", default=ModelType.GPT.value)
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--customer-model-size", default="base")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--customer-temperature", type=float, default=0.7)
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--persona", default="cold_chain_owner")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list-personas", action="store_true")
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    configure_stdio()
    args = parse_args()

    if args.list_personas:
        print(format_personas())
        return 0

    provider = parse_provider(args.provider)
    personas = list(PERSONAS.values()) if args.all else [choose_persona(args.persona)]

    try:
        for persona in personas:
            output_path = build_output_path(
                persona=persona,
                output=args.output,
                output_dir=args.output_dir,
                all_mode=args.all,
            )
            run_simulation(
                persona,
                provider=provider,
                config_path=args.config,
                model_size=args.model_size,
                customer_model_size=args.customer_model_size,
                temperature=args.temperature,
                customer_temperature=args.customer_temperature,
                turns_override=args.turns,
                output_path=output_path,
                user_id=args.user_id,
            )
    except Exception as exc:
        print(f"Simulation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
