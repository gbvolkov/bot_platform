from __future__ import annotations

import json

from .contracts import ContentItem, PublishManifest, ValidationReport


def build_report(
    *,
    items: list[ContentItem],
    validation_report: ValidationReport | None,
    manifest: PublishManifest | None,
) -> str:
    lines = ["# iSMART Template MVP Report", ""]
    lines.append(f"Items: {len(items)}")
    if validation_report:
        lines.append(f"Validation: {validation_report.status}")
        lines.append(f"Violations: {len(validation_report.violations)}")
        lines.append(f"Criteria: {len(validation_report.criteria)}")
    else:
        lines.append("Validation: not_run")
    if manifest:
        lines.append(f"Published items: {len(manifest.items)}")
    else:
        lines.append("Published items: 0")
    lines.append("")
    lines.append("| Content ID | Template | Status | Preview |")
    lines.append("| --- | --- | --- | --- |")
    for item in items:
        lines.append(
            f"| {item.content_id} | {item.template_id} | {item.status} | {item.preview_status} |"
        )
    if validation_report and validation_report.criteria:
        lines.append("")
        lines.append("## Validation Criteria")
        lines.append("")
        lines.append("| Criterion | Requirement | Status | Expected | Actual | Message |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for criterion in validation_report.criteria:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(criterion.criterion_id),
                        _cell(criterion.requirement_id),
                        _cell(criterion.status),
                        _cell(criterion.expected),
                        _cell(criterion.actual),
                        _cell(criterion.message),
                    ]
                )
                + " |"
            )
    if validation_report and validation_report.violations:
        lines.append("")
        lines.append("## Validation Issues")
        for violation in validation_report.violations:
            lines.append(f"- `{violation.code}` on `{violation.content_id}`: {violation.message}")
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        rendered = str(value)
    return rendered.replace("|", "\\|").replace("\n", "<br>")
