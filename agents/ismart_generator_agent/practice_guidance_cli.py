from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.utils import ModelType, get_llm

from .contracts import IsmartGenerationConfig
from .observability import build_callback_handlers
from .practice_guidance import run_practice_guidance_postprocess
from .subagents import build_subagent_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ismart-practice-guidance")
    parser.add_argument("--lesson-output", required=True, help="Path to one generated lesson output directory.")
    parser.add_argument("--prompts-dir", help="Prompt/skill directory. Defaults to the basic profile directory.")
    parser.add_argument("--max-generation-iterations", type=int, default=3)
    parser.add_argument("--max-reference-chars", type=int, default=0)
    parser.add_argument("--provider", default=ModelType.GPT.value, help="Model provider value or enum name.")
    parser.add_argument("--model-mode", choices=("base", "mini", "nano"), default="base")
    parser.add_argument("--no-llm-validator", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        provider = _parse_provider(args.provider)
        callbacks = build_callback_handlers(f"ismart_practice_guidance_{_timestamp()}")
        llm = get_llm(model=args.model_mode, provider=provider.value, temperature=0.2, streaming=False)
        subagents = build_subagent_registry(llm)
        config = IsmartGenerationConfig(
            prompts_dir=Path(args.prompts_dir) if args.prompts_dir else IsmartGenerationConfig().prompts_dir,
            output_root=Path(args.lesson_output),
            max_generation_iterations=args.max_generation_iterations,
            max_reference_chars=args.max_reference_chars,
            use_llm_validator=not args.no_llm_validator,
            verbose=bool(args.verbose),
            langchain_config={"callbacks": callbacks, "run_name": "ismart_practice_guidance"},
        )
        result = run_practice_guidance_postprocess(
            lesson_output_dir=Path(args.lesson_output),
            config=config,
            subagents=subagents,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failures.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "approved" else 1


def _parse_provider(value: str) -> ModelType:
    text = str(value or "").strip()
    for candidate in ModelType:
        if text.lower() == candidate.value.lower() or text.upper() == candidate.name:
            return candidate
    known = ", ".join(f"{item.name}/{item.value}" for item in ModelType)
    raise ValueError(f"Unknown provider {value!r}. Known providers: {known}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
