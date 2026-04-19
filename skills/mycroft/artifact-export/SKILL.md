---
name: artifact-export
description: Save, export, persist, or generate a downloadable GAZ sales result. Use only when the user explicitly asks to save, export, create a file, persist a recommendation, or produce a downloadable artifact.
---

# Artifact Export

Use this skill only on explicit save/export intent.

Reference documents:
- Direct tools: `../references/tools-and-actions.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- final user-facing content;
- requested format if any;
- artifact title or business purpose;
- whether execution details are requested or only final content.

## What to do

1. Confirm that the content is ready for export.
2. Export only the user-facing result, not internal traces, unless the user explicitly asks for a report with execution details.
3. Choose a concise title.
4. Use `store_artifact_tool`.
5. Preserve the returned link exactly.

## What to analyze

Check:
- whether the answer still needs source validation before being saved;
- whether exact facts in the exported content are confirmed;
- whether the requested artifact should include caveats and assumptions.

## Materials and tools

Use:
- `store_artifact_tool` for save/export;
- other subagents only if the content is not ready and the user asked to export a final recommendation.

## Output

Return the artifact link and a short description of what was saved.
