from __future__ import annotations

from .contracts import ContentItem, PublishManifest, PublishManifestItem
from .run_store import RunStore
from .validators import assert_publishable


def publish_items(store: RunStore, items: list[ContentItem]) -> PublishManifest:
    report = store.read_validation_report()
    assert_publishable(items, report)

    manifest_items: list[PublishManifestItem] = []
    for item in items:
        payload = build_sanitized_platform_payload(item)
        payload_ref = store.write_platform_payload(item.content_id, payload)
        item.status = "готово к отгрузке"
        store.update_item(item)
        manifest_items.append(
            PublishManifestItem(
                content_id=item.content_id,
                template_id=item.template_id,
                content_type=item.content_type,
                platform_payload=payload_ref,
                preview_required=item.preview_required,
                preview_status=item.preview_status,
            )
        )
    manifest = PublishManifest(request_id=items[0].request_id if items else "unknown", items=manifest_items)
    store.write_manifest(manifest)
    store.sync_hitl_from_items(items)
    return manifest


def build_sanitized_platform_payload(item: ContentItem) -> dict:
    return {
        "content_id": item.content_id,
        "course_id": item.course_id,
        "module_id": item.module_id,
        "lesson_id": item.lesson_id,
        "template_id": item.template_id,
        "content_type": item.content_type,
        "title": item.title,
        "payload": item.learner_payload,
        "preview_required": item.preview_required,
        "preview_status": item.preview_status,
        "preview_artifact": item.preview_artifact,
    }
