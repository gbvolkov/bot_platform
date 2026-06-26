from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config as root_config
from agents.utils import ModelType, extract_text, get_llm
from platform_utils.llm_logger import JSONFileTracer

from .context import material_result_summary, task_identity
from .contracts import IsmartGenerationConfig, IsmartGenerationResult
from .runtime import run_ismart_task
from .state import IsmartGeneratorAgentContext, IsmartGeneratorAgentState
from .subagents import build_subagent_registry
from .writer import safe_slug, write_batch_manifest


LOG = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = Path("docs") / "generated output"


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "ru",
    checkpoint_saver=None,
    *,
    streaming: bool = True,
    **_kwargs: Any,
):
    log_name = f"ismart_generator_agent_{time.strftime('%Y%m%d%H%M')}"
    callback_handlers = [JSONFileTracer(f"./logs/{log_name}")]
    if root_config.LANGFUSE_URL and len(root_config.LANGFUSE_URL) > 0:
        _ = Langfuse(
            public_key=root_config.LANGFUSE_PUBLIC,
            secret_key=root_config.LANGFUSE_SECRET,
            host=root_config.LANGFUSE_URL,
        )
        callback_handlers += [CallbackHandler()]

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    llm = get_llm(model="base", provider=provider.value, temperature=0.2, streaming=streaming)
    subagents = build_subagent_registry(llm)

    builder = StateGraph(IsmartGeneratorAgentState)
    builder.add_node("parse_request", create_parse_request_node())
    builder.add_node("run_generation", create_run_generation_node(subagents))
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "parse_request")
    builder.add_conditional_edges(
        "parse_request",
        route_after_parse,
        {
            "run_generation": "run_generation",
            "respond": "respond",
        },
    )
    builder.add_edge("run_generation", "respond")
    builder.add_edge("respond", END)

    return builder.compile(name="ismart_generator_agent", checkpointer=memory).with_config(
        {"callbacks": callback_handlers}
    )


def create_parse_request_node():
    def parse_request_node(
        state: IsmartGeneratorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IsmartGeneratorAgentContext],
    ) -> dict[str, Any]:
        try:
            context = _runtime_context(config, runtime)
            payload = _load_payload(context, state)
            tasks = filter_tasks(
                tasks_from_payload(payload),
                task_id=_optional_str(context.get("task_id")),
                lesson_number=_optional_str(context.get("lesson_number")),
            )
            return {"payload": payload, "tasks": tasks, "phase": "run_generation"}
        except Exception as exc:  # noqa: BLE001 - graph response should carry concise failures.
            LOG.exception("Failed to parse iSMART generator request")
            return {"error": str(exc), "phase": "respond"}

    return parse_request_node


def create_run_generation_node(subagents: Mapping[str, Any]):
    def run_generation_node(
        state: IsmartGeneratorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IsmartGeneratorAgentContext],
    ) -> dict[str, Any]:
        if state.get("error"):
            return {"phase": "respond"}
        try:
            context = _runtime_context(config, runtime)
            generation_config = _build_generation_config(context)
            results = run_tasks(state.get("tasks") or [], config=generation_config, subagents=subagents)
            public_results = [result.to_public_json() for result in results]
            return {
                "results": public_results,
                "output_text": format_agent_response(results),
                "phase": "respond",
            }
        except Exception as exc:  # noqa: BLE001 - graph response should carry concise failures.
            LOG.exception("Failed to run iSMART generator")
            return {"error": str(exc), "phase": "respond"}

    return run_generation_node


def route_after_parse(state: IsmartGeneratorAgentState) -> str:
    return "respond" if state.get("error") else "run_generation"


def respond_node(
    state: IsmartGeneratorAgentState,
    config: RunnableConfig,
    runtime: Runtime[IsmartGeneratorAgentContext],
) -> dict[str, Any]:
    if state.get("error"):
        content = f"iSMART generation failed: {state['error']}"
    else:
        content = state.get("output_text") or "iSMART generation finished."
    return {"messages": [AIMessage(content=content)], "phase": "done"}


def _runtime_context(
    config: RunnableConfig,
    runtime: Runtime[IsmartGeneratorAgentContext],
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if runtime.context:
        context.update(runtime.context)
    configurable = dict((config or {}).get("configurable") or {})
    for key in (
        "input",
        "input_url",
        "output",
        "task_id",
        "lesson_number",
        "max_generation_iterations",
        "max_package_repair_iterations",
        "max_reference_chars",
        "generation_target",
        "verbose",
    ):
        if key in configurable and key not in context:
            context[key] = configurable[key]
    return context


def _load_payload(context: dict[str, Any], state: IsmartGeneratorAgentState) -> Any:
    if context.get("input_url"):
        return load_payload_from_url(str(context["input_url"]))
    if context.get("input"):
        return load_payload_from_path_or_text(str(context["input"]))

    message_text = _last_human_text(state.get("messages") or [])
    if message_text:
        return load_payload_from_path_or_text(message_text)
    raise ValueError("Provide input JSON via context.input, context.input_url, or the latest human message.")


def _last_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            return extract_text(message).strip()
    return ""


def load_payload_from_path_or_text(value: str) -> Any:
    text = value.strip()
    path = _existing_path(text)
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(text)


def load_payload_from_url(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _existing_path(value: str) -> Path | None:
    try:
        path = Path(value)
        if path.exists() and path.is_file():
            return path
    except (OSError, ValueError):
        return None
    return None


def tasks_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_ensure_task(item) for item in payload]
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object, an array, or {'tasks': [...]}.")
    if isinstance(payload.get("tasks"), list):
        return [_ensure_task(item) for item in payload["tasks"]]
    if _is_single_task(payload):
        return [_ensure_task(payload)]
    if "course" in payload and isinstance(payload.get("modules"), list):
        return _tasks_from_course(payload)
    raise ValueError("Could not recognize input JSON shape.")


def filter_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_id: str | None = None,
    lesson_number: str | None = None,
) -> list[dict[str, Any]]:
    result = []
    for task in tasks:
        current_task_id, current_lesson_number, _ = task_identity(task)
        if task_id is not None and current_task_id != task_id:
            continue
        if lesson_number is not None and current_lesson_number != str(lesson_number):
            continue
        result.append(task)
    if (task_id or lesson_number) and not result:
        raise ValueError("No tasks matched selector.")
    return result


def run_tasks(
    tasks: list[dict[str, Any]],
    *,
    config: IsmartGenerationConfig,
    subagents: Mapping[str, Any],
) -> list[IsmartGenerationResult]:
    output_root = config.output_root
    if len(tasks) == 1:
        run_dir = output_root / f"run_{_timestamp()}_{safe_slug(task_identity(tasks[0])[0])}"
        if config.verbose:
            print(
                f"[ismart-generator-agent] single_task.start {json.dumps({'run_dir': str(run_dir)}, ensure_ascii=False)}",
                flush=True,
            )
        return [run_ismart_task(tasks[0], config, subagents=subagents, run_dir=run_dir)]

    batch_dir = output_root / f"batch_{_timestamp()}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    if config.verbose:
        print(
            f"[ismart-generator-agent] batch.start {json.dumps({'batch_dir': str(batch_dir), 'task_count': len(tasks)}, ensure_ascii=False)}",
            flush=True,
        )
    results: list[IsmartGenerationResult] = []
    module_summaries: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task in tasks:
        task_id, lesson_number, _ = task_identity(task)
        if config.verbose:
            print(
                f"[ismart-generator-agent] batch.task.start {json.dumps({'task_id': task_id, 'lesson_number': lesson_number}, ensure_ascii=False)}",
                flush=True,
            )
        module_key = str((task.get("module") or {}).get("title") or (task.get("lesson") or {}).get("module") or "")
        summaries = module_summaries.setdefault(module_key, {})
        run_dir = batch_dir / safe_slug(f"{lesson_number}-{task_id}")
        result = run_ismart_task(
            task,
            config,
            subagents=subagents,
            run_dir=run_dir,
            module_material_summaries=summaries,
        )
        results.append(result)
        if config.verbose:
            print(
                f"[ismart-generator-agent] batch.task.done {json.dumps({'task_id': task_id, 'status': result.status, 'output_dir': result.output_dir}, ensure_ascii=False)}",
                flush=True,
            )
        summaries[lesson_number] = [material_result_summary(material) for material in result.materials]
    write_batch_manifest(batch_dir, results)
    if config.verbose:
        print(
            f"[ismart-generator-agent] batch.done {json.dumps({'batch_dir': str(batch_dir)}, ensure_ascii=False)}",
            flush=True,
        )
    return results


def format_agent_response(results: list[IsmartGenerationResult]) -> str:
    if not results:
        return "iSMART generation finished: no tasks were selected."
    overall = "approved" if all(result.status == "approved" for result in results) else "has_failures"
    lines = [f"iSMART generation finished: {overall}", f"Tasks: {len(results)}"]
    for result in results:
        material_statuses = ", ".join(f"{item.kind}={item.status}" for item in result.materials)
        package_issues = len(result.package_validation.issues)
        lines.append(
            f"- lesson {result.lesson_number} ({result.task_id}): {result.status}; "
            f"materials: {material_statuses or 'none'}; package issues: {package_issues}; output: {result.output_dir}"
        )
    return "\n".join(lines)


def _build_generation_config(context: dict[str, Any]) -> IsmartGenerationConfig:
    output_root = Path(str(context.get("output") or DEFAULT_OUTPUT_ROOT))
    return IsmartGenerationConfig(
        output_root=output_root,
        max_generation_iterations=_int_context(context, "max_generation_iterations", 3),
        max_package_repair_iterations=_int_context(context, "max_package_repair_iterations", 2),
        max_reference_chars=_int_context(context, "max_reference_chars", 0),
        generation_target=_optional_str(context.get("generation_target")),
        verbose=bool(context.get("verbose", False)),
    )


def _int_context(context: dict[str, Any], key: str, default: int) -> int:
    value = context.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ensure_task(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not _is_single_task(value):
        raise ValueError("Each task must be an object with course, module, and lesson.")
    return value


def _is_single_task(value: dict[str, Any]) -> bool:
    return all(key in value for key in ("course", "module", "lesson"))


def _tasks_from_course(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    course = payload["course"]
    modules = payload.get("modules") or []
    for module in modules:
        for lesson in module.get("lessons") or []:
            tasks.append(
                {
                    "task_id": f"lesson-{lesson.get('lesson_number', len(tasks) + 1)}",
                    "course": course,
                    "module": module,
                    "lesson": lesson,
                    "modules": modules,
                    "markdown_references_base": payload.get("markdown_references_base"),
                }
            )
    return tasks


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
