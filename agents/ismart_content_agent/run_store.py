from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .contracts import (
    BatchGenerationRequest,
    ContentItem,
    GenerationRequest,
    HITLRecord,
    HITLState,
    PublishManifest,
    RunDocument,
    ValidationReport,
)


class RunStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_path = self.run_dir / "run.json"

    def initialize(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.run_path.exists():
            self._write_document(RunDocument())

    def write_input(self, request: GenerationRequest | BatchGenerationRequest) -> None:
        document = self._read_document()
        document.request = request
        self._write_document(document)

    def read_input(self) -> GenerationRequest | BatchGenerationRequest:
        document = self._read_document()
        if document.request is None:
            raise ValueError("run.json does not contain request")
        return document.request

    def write_item(self, item: ContentItem) -> None:
        document = self._read_document()
        document.items = [existing for existing in document.items if existing.content_id != item.content_id]
        document.items.append(item)
        self._write_document(document)

    def write_items(self, items: Iterable[ContentItem]) -> None:
        document = self._read_document()
        document.items = list(items)
        self._write_document(document)

    def read_items(self) -> list[ContentItem]:
        return self._read_document().items

    def update_item(self, item: ContentItem) -> None:
        self.write_item(item)

    def write_validation_report(self, report: ValidationReport) -> None:
        document = self._read_document()
        document.validation = report
        self._write_document(document)

    def read_validation_report(self) -> ValidationReport | None:
        return self._read_document().validation

    def write_hitl(self, state: HITLState) -> None:
        document = self._read_document()
        document.hitl = state
        self._write_document(document)

    def read_hitl(self) -> HITLState:
        document = self._read_document()
        if document.hitl is not None:
            return document.hitl
        request = self.read_input()
        return HITLState(request_id=request.request_id)

    def sync_hitl_from_items(self, items: list[ContentItem]) -> HITLState:
        request_id = items[0].request_id if items else self.read_input().request_id
        state = HITLState(request_id=request_id)
        for item in items:
            state.items[item.content_id] = HITLRecord(
                content_id=item.content_id,
                status=item.status,
                preview_required=item.preview_required,
                preview_status=item.preview_status,
                preview_artifact=item.preview_artifact,
            )
        self.write_hitl(state)
        return state

    def write_platform_payload(self, content_id: str, payload: dict) -> str:
        document = self._read_document()
        document.platform_payloads[content_id] = payload
        self._write_document(document)
        return f"run.json#/platform_payloads/{content_id}"

    def write_manifest(self, manifest: PublishManifest) -> None:
        document = self._read_document()
        document.manifest = manifest
        self._write_document(document)

    def read_manifest(self) -> PublishManifest | None:
        return self._read_document().manifest

    def read_document(self) -> RunDocument:
        return self._read_document()

    def write_report(self, text: str) -> None:
        (self.run_dir / "report.md").write_text(text, encoding="utf-8")

    def _read_document(self) -> RunDocument:
        if not self.run_path.exists():
            return RunDocument()
        return RunDocument.model_validate(json.loads(self.run_path.read_text(encoding="utf-8")))

    def _write_document(self, document: RunDocument) -> None:
        self.run_path.write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
