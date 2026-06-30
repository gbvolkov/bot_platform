from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.utils import ModelType, extract_text, get_llm

from .context import material_result_summary, task_identity
from .contracts import IsmartGenerationConfig, IsmartGenerationResult
from .observability import build_callback_handlers, langchain_config_from_runnable
from .profiles import resolve_course_level
from .runtime import run_ismart_task
from .state import IsmartGeneratorAgentContext, IsmartGeneratorAgentState
from .subagents import build_subagent_registry
from .task_skip import build_skipped_result, practice_task_count, skip_reason_for_task
from .writer import safe_slug, write_batch_manifest, write_task_output


LOG = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = Path("docs") / "generated output"


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "ru",
    checkpoint_saver=None,
    *,
    model_mode: str = "base",
    streaming: bool = True,
    **_kwargs: Any,
):
    log_name = f"ismart_generator_agent_{time.strftime('%Y%m%d%H%M')}"
    callback_handlers = build_callback_handlers(log_name)

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()

    def subagent_factory() -> Mapping[str, Any]:
        llm = get_llm(model=model_mode, provider=provider.value, temperature=0.2, streaming=streaming)
        return build_subagent_registry(llm)

    builder = StateGraph(IsmartGeneratorAgentState)
    builder.add_node("parse_request", create_parse_request_node())
    builder.add_node("run_generation", create_run_generation_node(subagent_factory))
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


def create_run_generation_node(subagent_factory: Callable[[], Mapping[str, Any]]):
    def run_generation_node(
        state: IsmartGeneratorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IsmartGeneratorAgentContext],
    ) -> dict[str, Any]:
        if state.get("error"):
            return {"phase": "respond"}
        try:
            context = _runtime_context(config, runtime)
            generation_config = _build_generation_config(
                context,
                langchain_config=langchain_config_from_runnable(config),
            )
            results = run_tasks(
                state.get("tasks") or [],
                config=generation_config,
                subagent_factory=subagent_factory,
            )
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
    subagents: Mapping[str, Any] | None = None,
    subagent_factory: Callable[[], Mapping[str, Any]] | None = None,
) -> list[IsmartGenerationResult]:
    if subagents is None and subagent_factory is None:
        raise ValueError("Either subagents or subagent_factory must be provided.")
    output_root = config.output_root
    if len(tasks) == 1:
        run_dir = output_root / f"run_{_timestamp()}_{safe_slug(task_identity(tasks[0])[0])}"
        skip_reason = skip_reason_for_task(tasks[0])
        if skip_reason:
            result = build_skipped_result(task=tasks[0], output_dir=run_dir, reason=skip_reason)
            write_task_output(
                result=result,
                output_dir=run_dir,
                validation_reports={"package": result.package_validation},
            )
            if config.verbose:
                task_id, lesson_number, lesson_title = task_identity(tasks[0])
                course_level = resolve_course_level(tasks[0])
                print(
                    f"[ismart-generator-agent] single_task.skipped {json.dumps({'task_id': task_id, 'lesson_number': lesson_number, 'lesson_title': lesson_title, 'course_level': course_level, 'resolved_profile': course_level, 'run_dir': str(run_dir), 'practice_task_count': practice_task_count(tasks[0]), 'reason': skip_reason}, ensure_ascii=False)}",
                    flush=True,
                )
            return [result]
        task_subagents = _build_task_subagents(subagents=subagents, subagent_factory=subagent_factory)
        if config.verbose:
            course_level = resolve_course_level(tasks[0])
            print(
                f"[ismart-generator-agent] single_task.start {json.dumps({'run_dir': str(run_dir), 'course_level': course_level, 'resolved_profile': course_level}, ensure_ascii=False)}",
                flush=True,
            )
            print(
                f"[ismart-generator-agent] single_task.subagents.reset {json.dumps({'run_dir': str(run_dir)}, ensure_ascii=False)}",
                flush=True,
            )
        return [run_ismart_task(tasks[0], config, subagents=task_subagents, run_dir=run_dir)]

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
        course_level = resolve_course_level(task)
        if config.verbose:
            print(
                f"[ismart-generator-agent] batch.task.start {json.dumps({'task_id': task_id, 'lesson_number': lesson_number, 'course_level': course_level, 'resolved_profile': course_level}, ensure_ascii=False)}",
                flush=True,
            )
        module_key = str((task.get("module") or {}).get("title") or (task.get("lesson") or {}).get("module") or "")
        summaries = module_summaries.setdefault(module_key, {})
        run_dir = batch_dir / safe_slug(f"{lesson_number}-{task_id}")
        skip_reason = skip_reason_for_task(task)
        if skip_reason:
            result = build_skipped_result(task=task, output_dir=run_dir, reason=skip_reason)
            write_task_output(
                result=result,
                output_dir=run_dir,
                validation_reports={"package": result.package_validation},
            )
            results.append(result)
            if config.verbose:
                print(
                    f"[ismart-generator-agent] batch.task.skipped {json.dumps({'task_id': task_id, 'lesson_number': lesson_number, 'course_level': course_level, 'resolved_profile': course_level, 'output_dir': str(run_dir), 'practice_task_count': practice_task_count(task), 'reason': skip_reason}, ensure_ascii=False)}",
                    flush=True,
                )
            continue
        task_subagents = _build_task_subagents(subagents=subagents, subagent_factory=subagent_factory)
        if config.verbose:
            print(
                f"[ismart-generator-agent] batch.task.subagents.reset {json.dumps({'task_id': task_id, 'lesson_number': lesson_number}, ensure_ascii=False)}",
                flush=True,
            )
        result = run_ismart_task(
            task,
            config,
            subagents=task_subagents,
            run_dir=run_dir,
            module_material_summaries=summaries,
        )
        results.append(result)
        if config.verbose:
            print(
                f"[ismart-generator-agent] batch.task.done {json.dumps({'task_id': task_id, 'course_level': result.course_level, 'resolved_profile': result.course_level, 'status': result.status, 'output_dir': result.output_dir}, ensure_ascii=False)}",
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


def _build_task_subagents(
    *,
    subagents: Mapping[str, Any] | None,
    subagent_factory: Callable[[], Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    if subagent_factory is not None:
        return subagent_factory()
    if subagents is not None:
        return subagents
    raise ValueError("Either subagents or subagent_factory must be provided.")


def format_agent_response(results: list[IsmartGenerationResult]) -> str:
    if not results:
        return "iSMART generation finished: no tasks were selected."
    overall = _overall_response_status(results)
    lines = [f"iSMART generation finished: {overall}", f"Tasks: {len(results)}"]
    for result in results:
        material_statuses = ", ".join(f"{item.kind}={item.status}" for item in result.materials)
        package_issues = len(result.package_validation.issues)
        skip_suffix = f"; skip reason: {result.skip_reason}" if result.skip_reason else ""
        lines.append(
            f"- lesson {result.lesson_number} ({result.task_id}, {result.course_level}): {result.status}; "
            f"materials: {material_statuses or 'none'}; package issues: {package_issues}; output: {result.output_dir}{skip_suffix}"
        )
    return "\n".join(lines)


def _overall_response_status(results: list[IsmartGenerationResult]) -> str:
    if any(result.status not in {"approved", "skipped"} for result in results):
        return "has_failures"
    if any(result.status == "skipped" for result in results):
        return "completed_with_skips"
    return "approved"


def _build_generation_config(
    context: dict[str, Any],
    *,
    langchain_config: dict[str, Any] | None = None,
) -> IsmartGenerationConfig:
    output_root = Path(str(context.get("output") or DEFAULT_OUTPUT_ROOT))
    return IsmartGenerationConfig(
        output_root=output_root,
        max_generation_iterations=_int_context(context, "max_generation_iterations", 3),
        max_package_repair_iterations=_int_context(context, "max_package_repair_iterations", 2),
        max_reference_chars=_int_context(context, "max_reference_chars", 0),
        generation_target=_optional_str(context.get("generation_target")),
        verbose=bool(context.get("verbose", False)),
        langchain_config=langchain_config or {},
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
