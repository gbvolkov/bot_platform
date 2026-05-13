from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from platform_guardrails.middleware import ToolExecutionSafetyMiddleware
from platform_guardrails.scanners import (
    LLMGuardScannerProfile,
    LLMGuardScannerRail,
    PROMPT_INJECTION_SENTENCE_PLACEHOLDER,
    ScannerSpec,
)
from platform_guardrails.tool_policy import ToolPolicyRail, ToolResultPolicy, ToolSecurityProfile
from test2 import (
    PROMPT_INJECTION_MODEL_CONFIG,
    PROMPT_INJECTION_THRESHOLD,
    TEXTS,
    print_exact_prompt_injection_sentences,
)


TOOL_NAME = "web_search"
SAMPLE_TEXT_KEY = "tool_with_injecttion"


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        execution_info=None,
        config={
            "configurable": {
                "tenant_id": "tenant",
                "user_id": "user",
                "thread_id": "tool-result-redaction-sample",
                "user_role": "default",
                "allow_external_tool_access": True,
            }
        },
    )


def _request() -> ToolCallRequest:
    tool_call = {
        "name": TOOL_NAME,
        "args": {"query": "улица Плющиха известные жители"},
        "id": "call-1",
        "type": "tool_call",
    }
    ai_message = AIMessage(content="", id="ai-tool-1", tool_calls=[tool_call])
    return ToolCallRequest(
        tool_call=tool_call,
        tool=None,
        state={"messages": [ai_message]},
        runtime=_runtime(),
    )


def _scanner_rail() -> LLMGuardScannerRail:
    return LLMGuardScannerRail(
        LLMGuardScannerProfile(
            input_scanners=(
                ScannerSpec(
                    "PromptInjection",
                    {
                        "threshold": PROMPT_INJECTION_THRESHOLD,
                        "match_type": "sentence",
                        "model": PROMPT_INJECTION_MODEL_CONFIG,
                    },
                ),
            ),
        )
    )


def _middleware(scanner_rail: LLMGuardScannerRail) -> ToolExecutionSafetyMiddleware:
    profile = ToolSecurityProfile(
        name=TOOL_NAME,
        allowed_roles=("default",),
        side_effect="external",
        category="external_access",
        allow_external_access=True,
        result_policy=ToolResultPolicy(scan_result=True, max_text_chars=100_000),
    )
    return ToolExecutionSafetyMiddleware(
        policy_rail=ToolPolicyRail([profile]),
        scanner_rail=scanner_rail,
        agent_name="test3.tool_result_redaction_sample",
    )


def _print_placeholder_excerpt(text: str) -> None:
    index = text.find(PROMPT_INJECTION_SENTENCE_PLACEHOLDER)
    if index < 0:
        print("\n== sanitized excerpt ==")
        print("No guarded sentence placeholder was inserted.")
        return

    start = max(0, index - 500)
    end = min(len(text), index + len(PROMPT_INJECTION_SENTENCE_PLACEHOLDER) + 500)
    print("\n== sanitized excerpt around guarded sentence ==")
    print(text[start:end])


def main() -> int:
    tool_result = TEXTS[SAMPLE_TEXT_KEY]

    print(f"== sample source: test2.TEXTS[{SAMPLE_TEXT_KEY!r}] ==")
    print(f"source_chars={len(tool_result)} threshold={PROMPT_INJECTION_THRESHOLD}")
    print_exact_prompt_injection_sentences(tool_result)

    middleware = _middleware(_scanner_rail())
    result = middleware.wrap_tool_call(
        _request(),
        lambda _request: ToolMessage(content=tool_result, tool_call_id="call-1"),
    )

    print("\n== guarded tool execution result ==")
    if isinstance(result, ToolMessage):
        print("continued=True")
        print(f"sanitized_chars={len(result.content)}")
        print(f"placeholder_count={result.content.count(PROMPT_INJECTION_SENTENCE_PLACEHOLDER)}")
        _print_placeholder_excerpt(result.content)
        print(f"\n\nTEXT:=====================================\n{result.content}\n======================================================\n\n")
        return 0

    print("continued=False")
    if isinstance(result, Command):
        print(result)
    else:
        print(type(result).__name__)
        print(result)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
