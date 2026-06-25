from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import IsmartGenerationConfig, ReferenceBundle, ReferenceDocument, repo_root
from .registry import REFERENCE_FIELDS


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def stable_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def read_prompt_files(config: IsmartGenerationConfig, prompt_files: tuple[str, ...]) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for name in prompt_files:
        path = config.prompts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt/skill file not found: {path}")
        prompts[name] = path.read_text(encoding="utf-8")
    return prompts


def reference_summary(bundle: ReferenceBundle) -> dict[str, list[dict[str, Any]]]:
    return {
        field: [document.to_public_json(include_content=False) for document in documents]
        for field, documents in bundle.items()
    }


class ReferenceLoader:
    def __init__(self, config: IsmartGenerationConfig) -> None:
        self.config = config

    def load(self, task: dict[str, Any]) -> ReferenceBundle:
        lesson = task.get("lesson") or {}
        materials_md = lesson.get("materials_md") or {}
        bundle: ReferenceBundle = {field: [] for field in REFERENCE_FIELDS}
        seen: set[Path] = set()
        for field in REFERENCE_FIELDS:
            for raw_path in materials_md.get(field) or []:
                resolved = self._resolve_reference_path(str(raw_path), task)
                if resolved in seen:
                    continue
                seen.add(resolved)
                content = resolved.read_text(encoding="utf-8")
                truncated = False
                if self.config.max_reference_chars and len(content) > self.config.max_reference_chars:
                    content = content[: self.config.max_reference_chars]
                    truncated = True
                bundle[field].append(
                    ReferenceDocument(
                        field=field,
                        path=str(raw_path),
                        resolved_path=str(resolved),
                        sha=stable_sha(content),
                        truncated=truncated,
                        content=content,
                    )
                )
        return bundle

    def _resolve_reference_path(self, raw_path: str, task: dict[str, Any]) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            resolved = candidate
        elif raw_path.replace("\\", "/").startswith("docs/"):
            resolved = repo_root() / candidate
        else:
            base_value = task.get("markdown_references_base")
            if base_value:
                base = Path(str(base_value))
                if not base.is_absolute():
                    base = repo_root() / base
                resolved = base / candidate
            else:
                resolved = self.config.prompts_dir.parent / candidate
        resolved = resolved.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Markdown reference not found: {raw_path} -> {resolved}")
        if resolved.suffix.lower() != ".md":
            raise ValueError(f"Reference is not Markdown: {raw_path}")
        return resolved
