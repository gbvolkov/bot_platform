from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.utils import ModelType, get_llm

from .context import task_identity
from .contracts import IsmartGenerationConfig, IsmartGenerationResult
from .observability import build_callback_handlers, langchain_config_from_runnable
from .profiles import resolve_course_level
from .python_sandbox import DisabledPythonSandbox, PythonSandbox
from .runtime import run_ismart_task
from .state import GeneratorRunRequest, IsmartGeneratorAgentContext, IsmartGeneratorAgentState
from .subagents import build_subagent_registry
from .task_skip import SKIPPED_MATERIAL_STATUSES
from .writer import safe_slug, write_batch_manifest, write_json


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

    llm = get_llm(model=model_mode, provider=provider.value, temperature=0.2, streaming=streaming)
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
            context = _request_context(state, config, runtime)
            payload = _load_payload(context, state)
            all_tasks = tasks_from_payload(payload)
            task_records = select_task_records(all_tasks, context)
            tasks = [record["task"] for record in task_records]
            return {"payload": payload, "tasks": tasks, "task_records": task_records, "phase": "run_generation"}
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
            context = _request_context(state, config, runtime)
            generation_config = _build_generation_config(
                context,
                langchain_config=langchain_config_from_runnable(config),
            )
            run_payload = run_task_records(
                state.get("task_records") or [
                    {"task": task, "source_index": index}
                    for index, task in enumerate(state.get("tasks") or [], start=1)
                ],
                config=generation_config,
                subagents=subagents,
                run_name=_optional_str(context.get("run_name")),
                preserve_source_index=bool(context.get("preserve_source_index", False)),
                stop_on_error=bool(context.get("stop_on_error", False)),
                stop_on_failure=bool(context.get("stop_on_failure", False)),
                write_sequential_manifest=bool(context.get("run_name") or context.get("preserve_source_index")),
            )
            results = run_payload["results"]
            public_results = [result.to_public_json() for result in results]
            return {
                "results": public_results,
                "task_entries": run_payload.get("task_entries") or [],
                "batch_manifest_path": run_payload.get("manifest_path"),
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
        "prompts_dir",
        "generation_target",
        "verbose",
        "run_name",
        "preserve_source_index",
        "stop_on_error",
        "stop_on_failure",
        "from_lesson",
        "to_lesson",
        "limit",
        "lesson_numbers",
        "task_ids",
        "previous_lessons_context",
        "previous_lesson_context",
        "resume_mode",
        "existing_output_root",
        "existing_lesson_output_dir",
        "existing_package",
    ):
        if key in configurable and key not in context:
            context[key] = configurable[key]
    return context


def _request_context(
    state: IsmartGeneratorAgentState,
    config: RunnableConfig,
    runtime: Runtime[IsmartGeneratorAgentContext],
) -> dict[str, Any]:
    return _runtime_context(config, runtime)


def _load_payload(context: dict[str, Any], state: IsmartGeneratorAgentState) -> Any:
    if context.get("input_url"):
        return load_payload_from_url(str(context["input_url"]))
    if context.get("input"):
        return load_payload_from_path_or_text(str(context["input"]))

    raise ValueError("Provide input JSON via runtime.context input or input_url.")


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


def select_task_records(tasks: list[dict[str, Any]], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    lesson_numbers = _string_set(context.get("lesson_numbers"))
    task_ids = _string_set(context.get("task_ids"))
    single_lesson = _optional_str(context.get("lesson_number"))
    single_task = _optional_str(context.get("task_id"))
    if single_lesson:
        lesson_numbers.add(single_lesson)
    if single_task:
        task_ids.add(single_task)
    from_lesson = _optional_int(context.get("from_lesson"))
    to_lesson = _optional_int(context.get("to_lesson"))
    limit = _optional_int(context.get("limit"))

    records: list[dict[str, Any]] = []
    for source_index, task in enumerate(tasks, start=1):
        task_id, lesson_number, _ = task_identity(task)
        if lesson_numbers and lesson_number not in lesson_numbers:
            continue
        if task_ids and task_id not in task_ids:
            continue
        lesson_as_int = _optional_int(lesson_number)
        if from_lesson is not None and (lesson_as_int is None or lesson_as_int < from_lesson):
            continue
        if to_lesson is not None and (lesson_as_int is None or lesson_as_int > to_lesson):
            continue
        records.append({"task": task, "source_index": source_index})
        if limit is not None and len(records) >= limit:
            break

    if (lesson_numbers or task_ids or from_lesson is not None or to_lesson is not None) and not records:
        raise ValueError("No tasks matched selector.")
    return records


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
    python_sandbox = _build_python_sandbox(config)
    try:
        if len(tasks) == 1:
            run_dir = output_root / f"run_{_timestamp()}_{safe_slug(task_identity(tasks[0])[0])}"
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
            return [
                run_ismart_task(
                    tasks[0],
                    config,
                    subagents=task_subagents,
                    run_dir=run_dir,
                    python_sandbox=python_sandbox,
                )
            ]

        batch_dir = output_root / f"batch_{_timestamp()}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        if config.verbose:
            print(
                f"[ismart-generator-agent] batch.start {json.dumps({'batch_dir': str(batch_dir), 'task_count': len(tasks)}, ensure_ascii=False)}",
                flush=True,
            )
        results: list[IsmartGenerationResult] = []
        for task in tasks:
            task_id, lesson_number, _ = task_identity(task)
            course_level = resolve_course_level(task)
            if config.verbose:
                print(
                    f"[ismart-generator-agent] batch.task.start {json.dumps({'task_id': task_id, 'lesson_number': lesson_number, 'course_level': course_level, 'resolved_profile': course_level}, ensure_ascii=False)}",
                    flush=True,
                )
            run_dir = batch_dir / safe_slug(f"{lesson_number}-{task_id}")
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
                python_sandbox=python_sandbox,
            )
            results.append(result)
            if config.verbose:
                print(
                    f"[ismart-generator-agent] batch.task.done {json.dumps({'task_id': task_id, 'course_level': result.course_level, 'resolved_profile': result.course_level, 'status': result.status, 'output_dir': result.output_dir}, ensure_ascii=False)}",
                    flush=True,
                )
        write_batch_manifest(batch_dir, results)
        if config.verbose:
            print(
                f"[ismart-generator-agent] batch.done {json.dumps({'batch_dir': str(batch_dir)}, ensure_ascii=False)}",
                flush=True,
            )
        return results
    finally:
        python_sandbox.close()


def run_task_records(
    task_records: list[dict[str, Any]],
    *,
    config: IsmartGenerationConfig,
    subagents: Mapping[str, Any],
    run_name: str | None = None,
    preserve_source_index: bool = False,
    stop_on_error: bool = False,
    stop_on_failure: bool = False,
    write_sequential_manifest: bool = False,
) -> dict[str, Any]:
    output_root = config.output_root
    if run_name:
        batch_dir = output_root / run_name
        use_batch_dir = True
    elif len(task_records) > 1:
        batch_dir = output_root / f"batch_{_timestamp()}"
        use_batch_dir = True
    else:
        batch_dir = output_root
        use_batch_dir = False

    if use_batch_dir:
        batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "sequential_manifest.json" if write_sequential_manifest else None
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(batch_dir),
        "task_count": len(task_records),
        "selected_task_count": len(task_records),
        "tasks": [],
    }
    if manifest_path:
        _write_runner_manifest(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "event": "sequential.start",
                    "output_dir": str(batch_dir),
                    "task_count": len(task_records),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    results: list[IsmartGenerationResult] = []
    task_entries: list[dict[str, Any]] = []
    python_sandbox = _build_python_sandbox(config)
    try:
        for selected_index, record in enumerate(task_records, start=1):
            task = record["task"]
            source_index = int(record.get("source_index") or selected_index)
            output_index = source_index if preserve_source_index else selected_index
            task_id, lesson_number, lesson_title = task_identity(task)
            course_level = resolve_course_level(task)
            if use_batch_dir:
                run_dir = batch_dir / safe_slug(f"{output_index:03d}-{lesson_number}-{task_id}")
            else:
                run_dir = output_root / f"run_{_timestamp()}_{safe_slug(task_id)}"

            if config.verbose:
                print(
                    json.dumps(
                        {
                            "event": "task.start",
                            "index": output_index,
                            "task_id": task_id,
                            "lesson_number": lesson_number,
                            "lesson_title": lesson_title,
                            "course_level": course_level,
                            "resolved_profile": course_level,
                            "output_dir": str(run_dir),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            try:
                result = run_ismart_task(
                    task,
                    config,
                    subagents=subagents,
                    run_dir=run_dir,
                    python_sandbox=python_sandbox,
                )
                results.append(result)
                entry = _manifest_entry_from_result(output_index, result)
                task_entries.append(entry)
                if config.verbose or write_sequential_manifest:
                    print(
                        json.dumps(
                            {
                                "event": "task.done",
                                "index": output_index,
                                "task_id": task_id,
                                "lesson_number": lesson_number,
                                "course_level": result.course_level,
                                "resolved_profile": result.course_level,
                                "status": result.status,
                                "output_dir": result.output_dir,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    for material in result.materials:
                        if material.status in SKIPPED_MATERIAL_STATUSES:
                            print(
                                json.dumps(
                                    {
                                        "event": "task.material_skipped",
                                        "index": output_index,
                                        "task_id": task_id,
                                        "lesson_number": lesson_number,
                                        "course_level": result.course_level,
                                        "resolved_profile": result.course_level,
                                        "material_kind": material.kind,
                                        "material_status": material.status,
                                        "reason": _material_skip_reason(material),
                                        "output_dir": result.output_dir,
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                if stop_on_failure and result.status not in {"approved", "completed_with_skips"}:
                    if manifest_path:
                        manifest["tasks"] = task_entries
                        manifest["status"] = "stopped_on_failure"
                        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
                        _write_runner_manifest(manifest_path, manifest)
                    break
            except Exception as exc:  # noqa: BLE001 - batch graph should isolate per-task failures when requested.
                entry = {
                    "index": output_index,
                    "task_id": task_id,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "course_level": course_level,
                    "resolved_profile": course_level,
                    "status": "error",
                    "output_dir": str(run_dir),
                    "error": str(exc),
                }
                task_entries.append(entry)
                write_json(run_dir / "error.json", entry)
                if config.verbose or write_sequential_manifest:
                    print(
                        json.dumps(
                            {
                                "event": "task.error",
                                "index": output_index,
                                "task_id": task_id,
                                "lesson_number": lesson_number,
                                "course_level": course_level,
                                "resolved_profile": course_level,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if stop_on_error:
                    if manifest_path:
                        manifest["tasks"] = task_entries
                        manifest["status"] = "stopped_on_error"
                        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
                        _write_runner_manifest(manifest_path, manifest)
                    break
                if not write_sequential_manifest:
                    raise
            finally:
                if manifest_path:
                    manifest["tasks"] = task_entries
                    _write_runner_manifest(manifest_path, manifest)
    finally:
        python_sandbox.close()

    if use_batch_dir and not write_sequential_manifest:
        write_batch_manifest(batch_dir, results)
    if manifest_path:
        manifest["tasks"] = task_entries
        manifest["status"] = _overall_entries_status(task_entries)
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_runner_manifest(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "event": "sequential.done",
                    "status": manifest["status"],
                    "output_dir": str(batch_dir),
                    "manifest": str(manifest_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {
        "results": results,
        "task_entries": task_entries,
        "manifest_path": str(manifest_path) if manifest_path else None,
    }


def _build_python_sandbox(config: IsmartGenerationConfig) -> PythonSandbox | DisabledPythonSandbox:
    if config.use_python_sandbox:
        return PythonSandbox(config)
    return DisabledPythonSandbox()


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


def _manifest_entry_from_result(index: int, result: IsmartGenerationResult) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "index": index,
        "task_id": result.task_id,
        "lesson_number": result.lesson_number,
        "lesson_title": result.lesson_title,
        "course_level": result.course_level,
        "resolved_profile": result.course_level,
        "status": result.status,
        "output_dir": result.output_dir,
        "materials": [
            {
                "kind": material.kind,
                "status": material.status,
                "iterations": material.iterations,
                "validation_issues": list(material.validation_issues),
                **(
                    {"skip_reason": _material_skip_reason(material)}
                    if material.status in SKIPPED_MATERIAL_STATUSES
                    else {}
                ),
            }
            for material in result.materials
        ],
        "package_validation": {
            "approved": result.package_validation.approved,
            "issues": result.package_validation.issues,
        },
    }
    skipped_materials = [
        {
            "kind": material.kind,
            "status": material.status,
            "reason": _material_skip_reason(material),
        }
        for material in result.materials
        if material.status in SKIPPED_MATERIAL_STATUSES
    ]
    if skipped_materials:
        entry["skipped_materials"] = skipped_materials
    return entry


def _write_runner_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _update_runner_manifest_counts(manifest)
    write_json(path, manifest)


def _update_runner_manifest_counts(manifest: dict[str, Any]) -> None:
    entries = manifest.get("tasks") or []
    manifest["generated_count"] = sum(1 for entry in entries if entry.get("status") not in {"skipped", "error"})
    manifest["approved_count"] = sum(1 for entry in entries if entry.get("status") == "approved")
    manifest["skipped_count"] = sum(1 for entry in entries if entry.get("status") == "skipped")
    manifest["completed_with_skips_count"] = sum(1 for entry in entries if entry.get("status") == "completed_with_skips")
    manifest["skipped_material_count"] = sum(len(entry.get("skipped_materials") or []) for entry in entries)
    manifest["error_count"] = sum(1 for entry in entries if entry.get("status") == "error")
    manifest["failed_count"] = sum(
        1 for entry in entries if entry.get("status") not in {"approved", "skipped", "completed_with_skips", "error"}
    )


def _overall_entries_status(entries: list[dict[str, Any]]) -> str:
    if any(entry.get("status") == "error" for entry in entries):
        return "has_errors"
    if any(entry.get("status") not in {"approved", "skipped", "completed_with_skips"} for entry in entries):
        return "has_failures"
    if any(entry.get("status") in {"skipped", "completed_with_skips"} for entry in entries):
        return "completed_with_skips"
    return "approved"


def _material_skip_reason(material: MaterialResult) -> str:
    reason = (material.generation_artifacts or {}).get("skip_reason")
    if reason:
        return str(reason)
    return str(material.agent_notes[0]) if material.agent_notes else ""


def format_agent_response(results: list[IsmartGenerationResult]) -> str:
    if not results:
        return "iSMART generation finished: no tasks were selected."
    overall = _overall_response_status(results)
    lines = [f"iSMART generation finished: {overall}", f"Tasks: {len(results)}"]
    for result in results:
        material_statuses = ", ".join(f"{item.kind}={item.status}" for item in result.materials)
        package_issues = len(result.package_validation.issues)
        lines.append(
            f"- lesson {result.lesson_number} ({result.task_id}, {result.course_level}): {result.status}; "
            f"materials: {material_statuses or 'none'}; package issues: {package_issues}; output: {result.output_dir}"
        )
    return "\n".join(lines)


def _overall_response_status(results: list[IsmartGenerationResult]) -> str:
    if any(result.status not in {"approved", "skipped", "completed_with_skips"} for result in results):
        return "has_failures"
    if any(result.status in {"skipped", "completed_with_skips"} for result in results):
        return "completed_with_skips"
    return "approved"


def _build_generation_config(
    context: dict[str, Any],
    *,
    langchain_config: dict[str, Any] | None = None,
) -> IsmartGenerationConfig:
    output_root = Path(str(context.get("output") or DEFAULT_OUTPUT_ROOT))
    return IsmartGenerationConfig(
        prompts_dir=Path(str(context["prompts_dir"])) if context.get("prompts_dir") else IsmartGenerationConfig().prompts_dir,
        output_root=output_root,
        max_generation_iterations=_int_context(context, "max_generation_iterations", 3),
        max_package_repair_iterations=_int_context(context, "max_package_repair_iterations", 2),
        max_reference_chars=_int_context(context, "max_reference_chars", 0),
        generation_target=_optional_str(context.get("generation_target")),
        verbose=bool(context.get("verbose", False)),
        langchain_config=langchain_config or {},
        previous_lessons_context=_context_list_of_dicts(
            context.get("previous_lessons_context", context.get("previous_lesson_context"))
        ),
        resume_mode=_optional_str(context.get("resume_mode")),
        existing_output_root=Path(str(context["existing_output_root"])) if context.get("existing_output_root") else None,
        existing_lesson_output_dir=(
            Path(str(context["existing_lesson_output_dir"])) if context.get("existing_lesson_output_dir") else None
        ),
        existing_package=dict(context["existing_package"]) if isinstance(context.get("existing_package"), dict) else None,
    )


def _context_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if not isinstance(value, list):
        return [{"content": str(value)}]
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(dict(item))
        elif item is not None and str(item).strip():
            result.append({"content": str(item)})
    return result


def _int_context(context: dict[str, Any], key: str, default: int) -> int:
    value = context.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _string_set(value: Any) -> set[str]:
    if value is None or value == "":
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if str(item).strip()}
    return {str(value)}


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
