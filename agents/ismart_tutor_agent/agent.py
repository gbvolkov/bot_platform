from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config as cfg

from agents.state.state import ConfigSchema
from agents.utils import ModelType, get_llm, _extract_text
from platform_utils.llm_logger import JSONFileTracer

from .prompts import (
    ISMART_TUTOR_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from .state import (
    HintGenerationResponse,
    IsmartTutorAgentContext,
    IsmartTutorAgentState,
    PersonInfoExtraction,
    PersonProfile,
)

LOG = logging.getLogger(__name__)

_NOSOLOGY_PATH = Path("data") / "nosologies.json"

_IMAGE_PART_TYPES = {"image_url", "input_image"}


def _extract_image_parts(message: BaseMessage) -> List[Dict[str, Any]]:
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []

    parts: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        part_type = item.get("type")
        if part_type in _IMAGE_PART_TYPES:
            parts.append(item)
            continue
        if part_type == "image":
            url = item.get("url")
            if isinstance(url, str) and url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str) and image_url.get("url"):
                parts.append({"type": "image_url", "image_url": {"url": image_url["url"]}})
                continue
    return parts


def _attachments_to_image_parts(attachments: Any) -> List[Dict[str, Any]]:
    if not isinstance(attachments, list):
        return []

    parts: List[Dict[str, Any]] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str) or not path:
            continue
        content_type = item.get("content_type")
        mime = None
        if isinstance(content_type, str) and content_type.strip():
            mime = content_type.split(";", 1)[0].strip().lower()
        if not mime or not mime.startswith("image/"):
            guessed, _ = mimetypes.guess_type(path)
            if isinstance(guessed, str) and guessed.startswith("image/"):
                mime = guessed
        if not mime or not mime.startswith("image/"):
            continue

        try:
            data = Path(path).read_bytes()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Failed to read image attachment %s: %s", path, exc)
            continue
        b64 = base64.b64encode(data).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return parts


def _is_reset_requested(state: IsmartTutorAgentState) -> bool:
    messages = state.get("messages") or []
    if not messages:
        return False
    last = messages[-1]
    if last.type != "human":
        return False
    content = getattr(last, "content", None)
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    return isinstance(first, dict) and first.get("type") == "reset"


def _clean_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _clean_int(value: Any, *, min_value: int, max_value: int) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            return None
        try:
            number = int(digits)
        except ValueError:
            return None
    else:
        return None
    if number < min_value or number > max_value:
        return None
    return number


@lru_cache(maxsize=1)
def _load_nosologies() -> List[Dict[str, Any]]:
    try:
        raw = _NOSOLOGY_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to load nosologies from %s: %s", _NOSOLOGY_PATH, exc)
        return []
    items = payload.get("nosologies") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _allowed_nosology_types() -> List[str]:
    types: List[str] = []
    for item in _load_nosologies():
        value = _clean_string(item.get("type"))
        if value:
            types.append(value)
    return types


def _find_nosology_by_type(nosology_type: str) -> Optional[Dict[str, Any]]:
    for item in _load_nosologies():
        if item.get("type") == nosology_type:
            return item
    return None


def _format_nosology_options() -> str:
    lines: List[str] = []
    for item in _load_nosologies():
        nosology_type = _clean_string(item.get("type"))
        if not nosology_type:
            continue
        nosology_name = _clean_string(item.get("name")) or ""
        if nosology_name and nosology_name != nosology_type:
            lines.append(f"- {nosology_type} — {nosology_name}")
        else:
            lines.append(f"- {nosology_type}")
    return "\n".join(lines) if lines else "- (nosology list unavailable)"


def _merge_profile(existing: Optional[PersonProfile], incoming: Dict[str, Any]) -> PersonProfile:
    merged: Dict[str, Any] = dict(existing or {})
    name = _clean_string(incoming.get("name"))
    if name:
        merged["name"] = name

    age = _clean_int(incoming.get("age"), min_value=1, max_value=120)
    if age is not None:
        merged["age"] = age

    school_year = _clean_int(incoming.get("school_year"), min_value=1, max_value=20)
    if school_year is not None:
        merged["school_year"] = school_year

    nosology_type = _clean_string(incoming.get("nosology_type"))
    if nosology_type and nosology_type in set(_allowed_nosology_types()):
        merged["nosology_type"] = nosology_type

    return merged  # type: ignore[return-value]


def _profile_missing_fields(profile: Optional[PersonProfile]) -> List[str]:
    profile = profile or {}
    missing: List[str] = []
    if not _clean_string(profile.get("name")):
        missing.append("name")
    if profile.get("age") is None:
        missing.append("age")
    if profile.get("school_year") is None:
        missing.append("school_year")
    if not _clean_string(profile.get("nosology_type")):
        missing.append("nosology_type")
    return missing


def _profile_is_complete(profile: Optional[PersonProfile]) -> bool:
    return len(_profile_missing_fields(profile)) == 0


def _build_person_extractor(model):
    allowed = _allowed_nosology_types()
    allowed_line = ", ".join(allowed) if allowed else "(nosology list unavailable)"
    system_prompt = "\n".join(
    [
        "Ты извлекаешь структурированную информацию о профиле ученика из последнего сообщения пользователя.",
        "",
        "Правила:",
        "- Извлекай ТОЛЬКО то, что явно указано.",
        "- Если поле отсутствует — НЕ добавляй ключ.",
        "- nosology_type должен быть одним из разрешённых типов (точное совпадение), иначе не добавляй его.",
        "",
        f"Разрешённые типы нозологий: {allowed_line}",
    ]
    ).strip()
    return create_agent(
        model=model,
        tools=None,
        system_prompt=system_prompt,
        response_format=ProviderStrategy(schema=PersonInfoExtraction),
        state_schema=IsmartTutorAgentState,
        context_schema=IsmartTutorAgentContext,
    )


def _build_person_info_request(missing: List[str], *, profile: Optional[PersonProfile] = None) -> str:
    fields_map = {
        "name": "name",
        "age": "age",
        "school_year": "school year",
        "nosology_type": "nosology type",
    }
    missing_labels = [fields_map.get(field, field) for field in missing]
    lines: List[str] = [
        "Уточните информацию о студенте, прежде, чем мы сформируем подсказки.",
    ]
    profile = profile or {}
    known_lines: List[str] = []
    if _clean_string(profile.get("name")):
        known_lines.append(f"- имя: {profile.get('name')}")
    if profile.get("age") is not None:
        known_lines.append(f"- возраст: {profile.get('age')}")
    if profile.get("school_year") is not None:
        known_lines.append(f"- год обучения: {profile.get('school_year')}")
    if _clean_string(profile.get("nosology_type")):
        known_lines.append(f"- тип нозологии: {profile.get('nosology_type')}")
    if known_lines:
        lines.append("Вы уже указали:")
        lines.extend(known_lines)
        lines.append("")

    lines.append("Пожалуйста, укажите также:")
    lines.extend(f"- {label}" for label in missing_labels)
    if "nosology_type" in missing:
        lines.append("")
        lines.append("Выберите тип нозологии из списка:")
        lines.append(_format_nosology_options())
    return "\n".join(lines).strip()


def collect_person_info_node(
    state: IsmartTutorAgentState,
    config: RunnableConfig,
    runtime: Runtime[IsmartTutorAgentContext],
) -> IsmartTutorAgentState:
    context: Any = getattr(runtime, "context", None)
    if isinstance(context, dict):
        person_id = _clean_string(context.get("person_id"))
        if person_id:
            return state
    return state


def check_person_info_node(state: IsmartTutorAgentState, config: RunnableConfig) -> IsmartTutorAgentState:
    missing = _profile_missing_fields(state.get("person_profile"))
    state["needs_person_info"] = bool(missing)
    return state


def create_extract_person_info_node(model):
    extractor = _build_person_extractor(model)

    def extract_person_info_node(
        state: IsmartTutorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IsmartTutorAgentContext],
    ) -> IsmartTutorAgentState:
        user_text = _clean_string(state.get("last_user_text")) or ""
        if not user_text:
            return state

        try:
            extract_state: IsmartTutorAgentState = {
                "messages": [HumanMessage(content=user_text)],
                "last_user_text": user_text,
            }
            result: IsmartTutorAgentState = extractor.invoke(
                extract_state,
                config=config,
                context=getattr(runtime, "context", None),
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Person extraction failed: %s", exc)
            return state

        structured = result.get("structured_response") or {}
        if not isinstance(structured, dict):
            return state

        profile = _merge_profile(state.get("person_profile"), structured)
        state["person_profile"] = profile

        return state

    return extract_person_info_node


def ask_person_info_node(state: IsmartTutorAgentState, config: RunnableConfig) -> IsmartTutorAgentState:
    profile = state.get("person_profile") or {}
    missing = _profile_missing_fields(profile)
    state["needs_person_info"] = bool(missing)
    if missing:
        state["hint_raw"] = _build_person_info_request(missing, profile=profile)
    return state


def ask_task_node(state: IsmartTutorAgentState, config: RunnableConfig) -> IsmartTutorAgentState:
    state["needs_question"] = True
    state["hint_raw"] = "Пришлите, пожалуйста, описание задачи, к которой надо сформировать подсказку."
    return state


def _build_hint_generator(model):
    return create_agent(
        model=model,
        tools=None,
        system_prompt=ISMART_TUTOR_SYSTEM_PROMPT,
        response_format=ProviderStrategy(schema=HintGenerationResponse),
        state_schema=IsmartTutorAgentState,
        context_schema=IsmartTutorAgentContext,
    )


def init_node(state: IsmartTutorAgentState, config: RunnableConfig) -> IsmartTutorAgentState:
    messages = state.get("messages") or []
    if not messages:
        return state

    configuration = config.get("configurable", {}) or {}
    state["user_info"] = {
        "user_id": configuration.get("user_id"),
        "user_role": configuration.get("user_role", "default"),
    }

    if _is_reset_requested(state):
        all_msg_ids = [m.id for m in messages if getattr(m, "id", None)]
        state["messages"] = [RemoveMessage(id=mid) for mid in all_msg_ids]
        state["question"] = ""
        state["hint_raw"] = ""
        state["person_id"] = None
        state["person_profile"] = None
        state["profile_complete_at_turn_start"] = False
        state["pending_question"] = None
        state["last_user_text"] = None
        state["needs_person_info"] = False
        state["needs_question"] = False
        state["reset_done"] = True
        return state

    state["reset_done"] = False
    state["profile_complete_at_turn_start"] = _profile_is_complete(state.get("person_profile"))
    last_user = next((m for m in reversed(messages) if m.type == "human"), None)
    last_user_text = _extract_text(last_user) if last_user else ""
    state["last_user_text"] = last_user_text
    state["hint_raw"] = None
    state["needs_person_info"] = False
    state["needs_question"] = False
    return state


def create_generate_hint_node(model):
    generator = _build_hint_generator(model)

    def generate_hint_node(
        state: IsmartTutorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IsmartTutorAgentContext],
    ) -> IsmartTutorAgentState:
        source_messages = state.get("messages") or []
        last_user = next((m for m in reversed(source_messages) if getattr(m, "type", None) == "human"), None)
        image_parts = _extract_image_parts(last_user) if last_user else []
        if not image_parts:
            image_parts = _attachments_to_image_parts(state.get("attachments"))

        last_user_text = _clean_string(state.get("last_user_text")) or ""
        if not last_user_text and not image_parts:
            state["needs_question"] = True
            state["hint_raw"] = "Жду текст задания."
            return state

        state["needs_question"] = False

        extra_parts: List[str] = []
        user_role = (state.get("user_info") or {}).get("user_role") or "default"
        if user_role and user_role != "default":
            extra_parts.append(f"User role: {user_role}")

        profile = state.get("person_profile") or {}
        if _profile_is_complete(profile):
            profile_lines = [
                "Student profile:",
                f"- name: {profile.get('name')}",
                f"- age: {profile.get('age')}",
                f"- school year: {profile.get('school_year')}",
                f"- nosology type: {profile.get('nosology_type')}",
            ]
            extra_parts.append("\n".join(profile_lines))

            nosology_type = _clean_string(profile.get("nosology_type"))
            if nosology_type:
                details = _find_nosology_by_type(nosology_type) or {}
                instructions = details.get("instructions")
                if isinstance(instructions, list) and instructions:
                    instr_lines = [str(item).strip() for item in instructions if str(item).strip()]
                    if instr_lines:
                        extra_parts.append(
                            "Nosology instructions:\n" + "\n".join(f"- {line}" for line in instr_lines[:8])
                        )

        messages: List[BaseMessage] = []
        extra_system = "\n\n".join(part for part in extra_parts if part).strip()
        if extra_system:
            messages.append(SystemMessage(content=extra_system))

        question_text = last_user_text or "(No text provided. Use the attached image(s).)"
        user_prompt = USER_PROMPT_TEMPLATE.format(question=question_text)

        if image_parts:
            messages.append(HumanMessage(content=[{"type": "text", "text": user_prompt}, *image_parts]))
        else:
            messages.append(HumanMessage(content=user_prompt))

        try:
            hint_state: IsmartTutorAgentState = {"messages": messages}
            result: IsmartTutorAgentState = generator.invoke(
                hint_state,
                config=config,
                context=getattr(runtime, "context", None),
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Hint generation failed: %s", exc)
            state["hint_raw"] = "I couldn't generate a hint. Please rephrase the task and include any missing details."
            return state

        structured = result.get("structured_response") or {}
        hint = _clean_string(structured.get("hint")) if isinstance(structured, dict) else None
        state["hint_raw"] = hint or "I couldn't generate a hint. Please rephrase the task and include any missing details."
        return state

    return generate_hint_node


def format_node(state: IsmartTutorAgentState, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    text = (state.get("hint_raw") or "").strip()
    if not text:
        text = "I couldn't generate a hint. Please rephrase the task and include any missing details."
    return {"messages": [AIMessage(content=[{"type": "text", "text": text}])]}


def _route_after_init(state: IsmartTutorAgentState) -> str:
    return "await" if state.get("reset_done") else "collect_person_info"


def _route_after_check_person_info_initial(state: IsmartTutorAgentState) -> str:
    if state.get("needs_person_info"):
        return "extract_person_info"
    if state.get("profile_complete_at_turn_start"):
        return "generate_hint"
    return "ask_task"


def _route_after_check_person_info_after_extraction(state: IsmartTutorAgentState) -> str:
    if state.get("needs_person_info"):
        return "ask_person_info"
    if not state.get("profile_complete_at_turn_start"):
        return "ask_task"
    return "generate_hint"


def initialize_agent(
    provider: ModelType = ModelType.GPT_PERS,
    use_platform_store: bool = False,
    checkpoint_saver=None,
):
    log_name = f"ismart_tutor_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]

    if cfg.LANGFUSE_URL:
        try:
            Langfuse(
                public_key=cfg.LANGFUSE_PUBLIC,
                secret_key=cfg.LANGFUSE_SECRET,
                host=cfg.LANGFUSE_URL,
            )
            callback_handlers.append(CallbackHandler())
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Langfuse initialisation failed: %s", exc)

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    hint_llm = get_llm(model="base", provider=provider.value, temperature=0.3)
    extract_llm = get_llm(model="mini", provider=provider.value, temperature=0.0)

    builder = StateGraph(IsmartTutorAgentState, config_schema=ConfigSchema)
    builder.add_node("init", init_node)
    builder.add_node("collect_person_info", collect_person_info_node)
    builder.add_node("check_person_info_initial", check_person_info_node)
    builder.add_node("extract_person_info", create_extract_person_info_node(extract_llm))
    builder.add_node("check_person_info_after_extraction", check_person_info_node)
    builder.add_node("ask_person_info", ask_person_info_node)
    builder.add_node("ask_task", ask_task_node)
    builder.add_node("generate_hint", create_generate_hint_node(hint_llm))
    builder.add_node("format", format_node)

    builder.add_edge(START, "init")
    builder.add_conditional_edges(
        "init",
        _route_after_init,
        {
            "collect_person_info": "collect_person_info",
            "await": END,
        },
    )
    builder.add_edge("collect_person_info", "check_person_info_initial")
    builder.add_conditional_edges(
        "check_person_info_initial",
        _route_after_check_person_info_initial,
        {
            "extract_person_info": "extract_person_info",
            "ask_task": "ask_task",
            "generate_hint": "generate_hint",
        },
    )
    builder.add_edge("extract_person_info", "check_person_info_after_extraction")
    builder.add_conditional_edges(
        "check_person_info_after_extraction",
        _route_after_check_person_info_after_extraction,
        {
            "ask_person_info": "ask_person_info",
            "ask_task": "ask_task",
            "generate_hint": "generate_hint",
        },
    )
    builder.add_edge("ask_person_info", "format")
    builder.add_edge("ask_task", "format")
    builder.add_edge("generate_hint", "format")
    builder.add_edge("format", END)

    graph = builder.compile(name="ismart_tutor_agent", checkpointer=memory).with_config({"callbacks": callback_handlers})
    return graph


if __name__ == "__main__":
    graph = initialize_agent()
    print(graph.get_graph().draw_ascii())
