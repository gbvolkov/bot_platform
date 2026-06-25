from __future__ import annotations

from pathlib import Path

from .contracts import BatchGenerationRequest, GenerationRequest, utc_now_iso
from .generators import generate_content_item
from .publisher import publish_items
from .reports import build_report
from .run_store import RunStore
from .solution_catalog import batch_task_configuration
from .source_reader import (
    DEFAULT_COURSE_TRACKER,
    DEFAULT_DONORS,
    extract_donor_sources,
    list_task_specs,
    resolve_request_sources,
    resolve_source_path,
)
from .validators import validate_items


def generate_run(request_path: str | Path, out_dir: str | Path) -> Path:
    request = GenerationRequest.model_validate_json(Path(request_path).read_text(encoding="utf-8"))
    request = resolve_request_sources(request)
    store = RunStore(out_dir)
    store.initialize()
    store.write_input(request)
    item = generate_content_item(request)
    store.write_item(item)
    store.sync_hitl_from_items([item])
    store.write_report(build_report(items=[item], validation_report=None, manifest=None))
    return store.run_dir


def generate_batch_run(request_path: str | Path, out_dir: str | Path) -> Path:
    batch = BatchGenerationRequest.model_validate_json(Path(request_path).read_text(encoding="utf-8"))
    tracker_path = resolve_source_path(batch.course_tracker_path, DEFAULT_COURSE_TRACKER)
    donors_path = resolve_source_path(batch.donors_path, DEFAULT_DONORS)
    task_specs = list_task_specs(
        tracker_path=tracker_path,
        audience=batch.audience,
        limit=batch.first_task_count,
    )

    items = []
    for index, task in enumerate(task_specs, start=1):
        template_id, service_spec = batch_task_configuration(task)
        lesson_number = task["lesson_number"]
        donors = extract_donor_sources(donors_path=donors_path, lesson_number=lesson_number)
        request = GenerationRequest(
            request_id=f"{batch.request_id}-{index:03d}",
            course_id=batch.course_id,
            module_id=f"module-{task['module_number']}",
            lesson_id=f"lesson-{lesson_number}",
            template_id=template_id,
            topic=task["topic"],
            lesson_title=task["lesson_title"],
            learning_goal=f"Обучающийся выполняет задание из программы занятия: {task['task_text']}",
            audience=batch.audience,
            level=batch.level,
            lesson_number=lesson_number,
            target_task_level=task["task_level"],
            target_task_number=task["task_number"],
            task_spec=task,
            service_spec=service_spec,
            source_refs=[
                f"{task['source_document']}::{task['sheet_name']}::row {task['row_index']}",
                f"{DEFAULT_DONORS.name}::з{lesson_number}",
            ],
            donor_sources=donors,
            metadata={"batch_index": index, "batch_request_id": batch.request_id},
        )
        items.append(generate_content_item(request))

    store = RunStore(out_dir)
    store.initialize()
    store.write_input(batch)
    store.write_items(items)
    store.sync_hitl_from_items(items)
    store.write_report(build_report(items=items, validation_report=None, manifest=None))
    return store.run_dir


def validate_run(run_dir: str | Path) -> None:
    store = RunStore(run_dir)
    items = store.read_items()
    report = validate_items(items)
    if report.status == "passed":
        for item in items:
            item.status = "проверено"
            item.updated_at = utc_now_iso()
    else:
        for item in items:
            item.status = "на доработку"
            item.updated_at = utc_now_iso()
    store.write_items(items)
    store.write_validation_report(report)
    store.sync_hitl_from_items(items)
    store.write_report(build_report(items=items, validation_report=report, manifest=None))


def mark_approved(run_dir: str | Path, content_id: str) -> None:
    store = RunStore(run_dir)
    items = store.read_items()
    found = False
    for item in items:
        if item.content_id == content_id:
            found = True
            if item.status not in {"проверено", "одобрено"}:
                raise ValueError(f"Cannot approve item in status {item.status}")
            item.status = "одобрено"
            item.updated_at = utc_now_iso()
            store.update_item(item)
            break
    if not found:
        raise ValueError(f"Unknown content_id: {content_id}")
    store.sync_hitl_from_items(items)
    store.write_report(
        build_report(
            items=items,
            validation_report=store.read_validation_report(),
            manifest=None,
        )
    )


def mark_preview_passed(run_dir: str | Path, content_id: str, artifact: str) -> None:
    store = RunStore(run_dir)
    items = store.read_items()
    found = False
    for item in items:
        if item.content_id == content_id:
            found = True
            if not item.preview_required:
                raise ValueError(f"Content item {content_id} does not require preview")
            item.preview_status = "passed"
            item.preview_artifact = artifact
            item.updated_at = utc_now_iso()
            store.update_item(item)
            break
    if not found:
        raise ValueError(f"Unknown content_id: {content_id}")
    store.sync_hitl_from_items(items)
    store.write_report(
        build_report(
            items=items,
            validation_report=store.read_validation_report(),
            manifest=None,
        )
    )


def publish_run(run_dir: str | Path) -> None:
    store = RunStore(run_dir)
    items = store.read_items()
    manifest = publish_items(store, items)
    store.write_report(
        build_report(
            items=store.read_items(),
            validation_report=store.read_validation_report(),
            manifest=manifest,
        )
    )
