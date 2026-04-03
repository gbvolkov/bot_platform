from __future__ import annotations

import argparse
import asyncio
import json
import locale
import shlex
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

try:
    import termios
except ImportError:  # pragma: no cover - Windows
    termios = None

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    message_chunk_to_message,
)
from langchain_core.messages.base import message_to_dict

from agents.sales_lead_agent.agent import initialize_agent
from agents.utils import ModelType, extract_text


DEFAULT_PROVIDER = ModelType.GPT.value
HELP_TEXT = """
Commands:
  /scenario <name>  switch scripted scenario
  /export [path] [--no-tools]  export messages from current agent state
  /reset            start a fresh dialog session
  /help             show commands
  /exit             leave the simulator

Use /multiline ... /end for multi-line input.
""".strip()
_MULTILINE_START = "/multiline"
_MULTILINE_END = "/end"
_INPUT_FALLBACK_ENCODINGS = ("utf-8", "cp1251", "cp866", "latin-1")


def _configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


def _simulator_checkpointer_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "sales_lead_agent" / "simulator_checkpoints.sqlite"


def _simulator_session_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "sales_lead_agent" / "simulator_session.json"


def _simulator_export_root() -> Path:
    return Path(__file__).resolve().parent / "data" / "simulator_transcripts"


def _persistent_checkpoint_saver():
    checkpoint_path = _simulator_checkpointer_path()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(str(checkpoint_path))


@dataclass(frozen=True)
class ScenarioExpectation:
    answer_type: str | None = None
    required_tools: tuple[str, ...] = ()
    require_active_run: bool = False
    require_searchable_index: bool = False
    reuse_active_run_from_previous: bool = False
    require_document_evidence: bool = False
    require_document_fact_labels: bool = False
    require_visible_support: bool = False


@dataclass(frozen=True)
class ScenarioStep:
    user_text: str
    expectation: ScenarioExpectation = field(default_factory=ScenarioExpectation)


@dataclass(frozen=True)
class TurnOutcome:
    user_text: str
    reply: str
    result: dict[str, Any]
    thread_id: str
    agent_status: str = "completed"
    interrupt_payload: dict[str, Any] | None = None


SCENARIOS: dict[str, list[ScenarioStep]] = {
    "procurement_search": [
        ScenarioStep(
            "Find new relevant transport insurance procurements.",
            ScenarioExpectation(
                required_tools=("purchase_search_tool",),
                require_active_run=True,
            ),
        ),
    ],
    "procurement_analysis": [
        ScenarioStep(
            "Analyze one transport insurance procurement and summarize the main risks.",
            ScenarioExpectation(
                required_tools=("purchase_search_tool",),
                require_active_run=True,
            ),
        ),
    ],
    "company_check": [
        ScenarioStep(
            "Check company by INN 7707083893.",
            ScenarioExpectation(
                answer_type="company_check",
                required_tools=("counterparty_scoring_tool", "counterparty_fssp_tool"),
                require_visible_support=True,
            ),
        ),
    ],
    "fact_lookup": [
        ScenarioStep(
            "Show where the documents specify bidder experience requirements.",
            ScenarioExpectation(),
        ),
    ],
    "comparison": [
        ScenarioStep(
            "Compare companies 7707083893 and 7710140679 and tell me which one to pass to sales first.",
            ScenarioExpectation(
                answer_type="comparison",
                required_tools=("counterparty_scoring_tool", "counterparty_fssp_tool"),
                require_visible_support=True,
            ),
        ),
    ],
    "procurement_search_followup_fact_lookup": [
        ScenarioStep(
            "Find new relevant transport insurance procurements.",
            ScenarioExpectation(
                required_tools=("purchase_search_tool",),
                require_active_run=True,
            ),
        ),
        ScenarioStep(
            "Now show where the documents specify bidder experience requirements.",
            ScenarioExpectation(
                answer_type="lead_card",
                required_tools=("doc_search_tool",),
                require_active_run=True,
                require_searchable_index=True,
                reuse_active_run_from_previous=True,
                require_document_evidence=True,
                require_document_fact_labels=True,
            ),
        ),
    ],
    "procurement_analysis_followup_doc_source": [
        ScenarioStep(
            "Analyze one transport insurance procurement and summarize the main risks.",
            ScenarioExpectation(
                required_tools=("purchase_search_tool",),
                require_active_run=True,
            ),
        ),
        ScenarioStep(
            "Show the source snippet for the main risk.",
            ScenarioExpectation(
                answer_type="lead_card",
                require_active_run=True,
                require_searchable_index=True,
                reuse_active_run_from_previous=True,
                require_document_evidence=True,
                require_document_fact_labels=True,
            ),
        ),
    ],
    "company_check_then_comparison": [
        ScenarioStep(
            "Check company by INN 7707083893.",
            ScenarioExpectation(
                answer_type="company_check",
                required_tools=("counterparty_scoring_tool", "counterparty_fssp_tool"),
                require_visible_support=True,
            ),
        ),
        ScenarioStep(
            "Compare companies 7707083893 and 7710140679 and tell me which one to pass to sales first.",
            ScenarioExpectation(
                answer_type="comparison",
                required_tools=("counterparty_scoring_tool", "counterparty_fssp_tool"),
                require_visible_support=True,
            ),
        ),
    ],
    "comparison_then_prioritization_followup": [
        ScenarioStep(
            "Compare companies 7707083893 and 7710140679 and tell me which one to pass to sales first.",
            ScenarioExpectation(
                answer_type="comparison",
                required_tools=("counterparty_scoring_tool", "counterparty_fssp_tool"),
                require_visible_support=True,
            ),
        ),
        ScenarioStep(
            "Briefly explain why that company should be prioritized.",
            ScenarioExpectation(answer_type="comparison", require_visible_support=True),
        ),
    ],
    "repeated_same_request": [
        ScenarioStep(
            "Find new relevant transport insurance procurements.",
            ScenarioExpectation(
                required_tools=("purchase_search_tool",),
                require_active_run=True,
            ),
        ),
        ScenarioStep(
            "Find new relevant transport insurance procurements.",
            ScenarioExpectation(
                required_tools=("purchase_search_tool",),
                require_active_run=True,
                reuse_active_run_from_previous=True,
            ),
        ),
    ],
}


def _parse_provider(raw_value: str) -> ModelType:
    normalized = raw_value.strip().lower()
    for provider in ModelType:
        if normalized in {provider.value.lower(), provider.name.lower()}:
            return provider
    available = ", ".join(sorted(provider.value for provider in ModelType))
    raise ValueError(f"Unknown provider '{raw_value}'. Available values: {available}")


def _new_config() -> dict[str, Any]:
    return {"configurable": {"thread_id": f"sales-lead-sim-{uuid.uuid4().hex}"}}


def _load_persistent_interactive_session(*, default_scenario: str) -> tuple[dict[str, Any], str, bool]:
    session_path = _simulator_session_path()
    if not session_path.exists():
        return _new_config(), default_scenario, False
    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception:
        return _new_config(), default_scenario, False
    thread_id = str(payload.get("thread_id") or payload.get("conversation_id") or "").strip()
    scenario = str(payload.get("scenario") or "").strip() or default_scenario
    if not thread_id:
        return _new_config(), scenario, False
    return {"configurable": {"thread_id": thread_id}}, scenario, True


def _save_persistent_interactive_session(*, config: dict[str, Any], scenario: str) -> None:
    thread_id = str(config.get("configurable", {}).get("thread_id") or "").strip()
    if not thread_id:
        return
    session_path = _simulator_session_path()
    session_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conversation_id": thread_id,
        "thread_id": thread_id,
        "scenario": scenario,
    }
    session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _latest_ai_text(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages") or []):
        if isinstance(message, AIMessage):
            return extract_text(message)
    return ""


def _coerce_interrupt_payload(raw_value: Any) -> dict[str, Any] | None:
    payload = getattr(raw_value, "value", raw_value)
    return payload if isinstance(payload, dict) else None


def _extract_interrupt_payload(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return None
    return _coerce_interrupt_payload(interrupts[-1])


def _interrupt_question(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("question") or payload.get("content") or "").strip()


def _print_progress_event(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("type") != "progress":
        return
    message = str(payload.get("message") or "").strip()
    if not message:
        return
    print(f"[Progress] {message}")


def _decode_console_bytes(raw_value: bytes) -> str:
    encodings: list[str] = []
    seen: set[str] = set()
    for encoding in (
        sys.stdin.encoding,
        locale.getpreferredencoding(False),
        *_INPUT_FALLBACK_ENCODINGS,
    ):
        normalized = str(encoding or "").strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        encodings.append(normalized)
    for encoding in encodings:
        try:
            return raw_value.decode(encoding)
        except UnicodeDecodeError:
            continue
    fallback = encodings[0] if encodings else "utf-8"
    return raw_value.decode(fallback, errors="replace")


def _read_raw_console_input(prompt: str) -> str:
    is_tty = getattr(sys.stdin, "isatty", None)
    if callable(is_tty) and is_tty():
        return input(prompt)
    stdin_buffer = getattr(sys.stdin, "buffer", None)
    if stdin_buffer is None:
        return input(prompt)
    print(prompt, end="", flush=True)
    raw_value = stdin_buffer.readline()
    if not raw_value:
        raise EOFError
    return _decode_console_bytes(raw_value.rstrip(b"\r\n"))


def _read_interrupt_input(prompt: str) -> str:
    raw_value = _read_raw_console_input(prompt)
    if raw_value.strip().lower() in {"/exit", "exit"}:
        raise KeyboardInterrupt
    return raw_value


def _read_text_input(prompt: str, *, allow_multiline: bool = False) -> str:
    raw_value = _read_interrupt_input(prompt)
    if not allow_multiline or raw_value.strip().lower() != _MULTILINE_START:
        return raw_value
    print(f"[Multiline mode] Paste text. Finish with {_MULTILINE_END} on a separate line.")
    lines: list[str] = []
    while True:
        line = _read_interrupt_input("... ")
        if line.strip().lower() == _MULTILINE_END:
            return "\n".join(lines)
        lines.append(line)


def _handle_command(raw_text: str, *, scenario: str) -> tuple[bool, str, bool]:
    command = raw_text.strip()
    if not command.startswith("/"):
        return False, scenario, False
    lowered = command.lower()
    if lowered in {"/exit", "exit"}:
        raise KeyboardInterrupt
    if lowered == "/help":
        print(f"\n{HELP_TEXT}\n")
        return True, scenario, False
    if lowered == "/reset":
        return True, scenario, True
    if lowered.startswith("/scenario "):
        candidate = command[len("/scenario ") :].strip()
        if candidate not in SCENARIOS:
            print("\n[Simulator] unknown scenario.\n")
            return True, scenario, False
        print(f"\n[Simulator] scenario set to: {candidate}\n")
        return True, candidate, False
    print("\n[Simulator] unknown command. Use /help.\n")
    return True, scenario, False


def _parse_export_command(command: str) -> tuple[str | None, bool]:
    raw_args = command[len("/export") :].strip()
    if not raw_args:
        return None, False
    parts = shlex.split(raw_args, posix=False)
    no_tools = False
    remaining: list[str] = []
    for part in parts:
        if part == "--no-tools":
            no_tools = True
            continue
        remaining.append(part)
    target_path = " ".join(remaining).strip() or None
    return target_path, no_tools


def _default_export_path(*, thread_id: str, no_tools: bool = False) -> Path:
    safe_thread_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in thread_id).strip("_")
    suffix = "_state_messages_human_ai" if no_tools else "_state_messages"
    file_name = f"{safe_thread_id or 'dialog'}{suffix}.json"
    root = _simulator_export_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / file_name


def _markdown_export_path_for(json_path: Path) -> Path:
    return json_path.with_suffix(".md")


async def _get_state_snapshot(graph: Any, config: dict[str, Any]) -> Any:
    aget_state = getattr(graph, "aget_state", None)
    if callable(aget_state):
        return await aget_state(config)
    get_state = getattr(graph, "get_state", None)
    if callable(get_state):
        return get_state(config)
    return None


def _serialize_state_messages(messages: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        try:
            payload = message_to_dict(message)
        except Exception:
            payload = {
                "type": getattr(message, "type", message.__class__.__name__),
                "data": {
                    "content": extract_text(message) if hasattr(message, "content") else str(message),
                },
            }
        payload["index"] = index
        serialized.append(payload)
    return serialized


def _filter_export_messages(messages: list[Any], *, no_tools: bool) -> list[Any]:
    if not no_tools:
        return list(messages)
    return [message for message in messages if isinstance(message, (HumanMessage, AIMessage))]


def _message_content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        parts.append(text)
                    continue
                parts.append(json.dumps(item, ensure_ascii=False, indent=2))
            else:
                parts.append(str(item))
        return "\n\n".join(part for part in parts if part)
    if content is None:
        return ""
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def _message_heading(message_type: str) -> str:
    normalized = str(message_type or "").strip().lower()
    mapping = {
        "human": "Human",
        "ai": "AI",
        "tool": "Tool",
        "system": "System",
    }
    return mapping.get(normalized, normalized.title() or "Message")


def _render_state_messages_markdown(
    *,
    thread_id: str,
    state_keys: list[str],
    messages: list[dict[str, Any]],
    message_filter: str,
) -> str:
    lines = [
        f"# Agent State Export",
        "",
        f"- Thread ID: `{thread_id}`",
        f"- Message count: {len(messages)}",
        f"- Message filter: {message_filter}",
        f"- State keys: {', '.join(state_keys)}",
        "",
    ]
    for message in messages:
        index = message.get("index")
        message_type = str(message.get("type") or "")
        data = message.get("data") if isinstance(message.get("data"), dict) else {}
        lines.append(f"## {index}. {_message_heading(message_type)}")
        tool_name = str(data.get("name") or "").strip()
        tool_call_id = str(data.get("tool_call_id") or "").strip()
        if tool_name:
            lines.append(f"- Name: `{tool_name}`")
        if tool_call_id:
            lines.append(f"- Tool call id: `{tool_call_id}`")
        lines.append("")
        content = _message_content_as_text(data.get("content"))
        if content:
            lines.append(content)
        else:
            lines.append("(empty content)")
        additional_kwargs = data.get("additional_kwargs")
        if isinstance(additional_kwargs, dict) and additional_kwargs:
            lines.extend(
                [
                    "",
                    "### additional_kwargs",
                    "```json",
                    json.dumps(additional_kwargs, ensure_ascii=False, indent=2),
                    "```",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def _export_dialog_state(
    *,
    graph: Any,
    config: dict[str, Any],
    target_path: str | None = None,
    no_tools: bool = False,
) -> tuple[Path, Path]:
    snapshot = await _get_state_snapshot(graph, config)
    values = getattr(snapshot, "values", None)
    state = values if isinstance(values, dict) else {}
    messages = state.get("messages")
    message_list = _filter_export_messages(messages if isinstance(messages, list) else [], no_tools=no_tools)
    thread_id = str(config.get("configurable", {}).get("thread_id") or "").strip()
    export_path = Path(target_path).expanduser() if target_path else _default_export_path(thread_id=thread_id, no_tools=no_tools)
    if not export_path.is_absolute():
        export_path = Path.cwd() / export_path
    if export_path.suffix.lower() == ".md":
        markdown_path = export_path
        export_path = export_path.with_suffix(".json")
    else:
        if export_path.suffix.lower() != ".json":
            export_path = export_path.with_suffix(".json")
        markdown_path = _markdown_export_path_for(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    state_keys = sorted(state.keys())
    serialized_messages = _serialize_state_messages(message_list)
    payload = {
        "thread_id": thread_id,
        "message_filter": "human_ai" if no_tools else "all",
        "message_count": len(message_list),
        "state_keys": state_keys,
        "messages": serialized_messages,
    }
    export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        _render_state_messages_markdown(
            thread_id=thread_id,
            state_keys=state_keys,
            messages=serialized_messages,
            message_filter="human_ai" if no_tools else "all",
        ),
        encoding="utf-8",
    )
    return export_path, markdown_path


async def _handle_runtime_command(
    raw_text: str,
    *,
    graph: Any,
    config: dict[str, Any],
    scenario: str,
) -> tuple[bool, str, bool]:
    command = raw_text.strip()
    lowered = command.lower()
    if lowered == "/export" or lowered.startswith("/export "):
        target_path, no_tools = _parse_export_command(command)
        export_path, markdown_path = await _export_dialog_state(
            graph=graph,
            config=config,
            target_path=target_path,
            no_tools=no_tools,
        )
        label = "human/ai state messages" if no_tools else "state messages"
        print(
            f"\n[Simulator] {label} exported to:\n- JSON: {export_path}\n- MD:   {markdown_path}\n"
        )
        return True, scenario, False
    return _handle_command(raw_text, scenario=scenario)


async def _invoke_agent(
    graph: Any,
    config: dict[str, Any],
    user_text: str | None,
    *,
    resume_text: str | None = None,
) -> TurnOutcome:
    if resume_text is not None:
        payload: Any = Command(resume=resume_text)
    else:
        payload = {} if user_text is None else {"messages": [HumanMessage(content=user_text)]}
    result: dict[str, Any] | None = None
    merged_chunk: AIMessageChunk | None = None
    last_ai_message: AIMessage | None = None

    async for item in graph.astream(
        payload,
        config=config,
        stream_mode=["messages", "custom", "values"],
        subgraphs=True,
    ):
        if not isinstance(item, tuple):
            continue
        if len(item) == 2:
            mode, payload_item = item
        elif len(item) == 3:
            _namespace, mode, payload_item = item
        else:
            continue
        if mode == "messages":
            message, _meta = payload_item
            if isinstance(message, AIMessageChunk):
                merged_chunk = message if merged_chunk is None else merged_chunk + message
            elif isinstance(message, AIMessage):
                last_ai_message = message
                if merged_chunk is not None:
                    merged_message = message_chunk_to_message(merged_chunk)
                    if extract_text(merged_message) == extract_text(message):
                        merged_chunk = None
        elif mode == "custom":
            _print_progress_event(payload_item)
        elif mode == "values" and isinstance(payload_item, dict):
            result = payload_item

    if result is None and last_ai_message is None and merged_chunk is not None:
        last_ai_message = message_chunk_to_message(merged_chunk)
    if result is None and last_ai_message is not None:
        result = {"messages": [last_ai_message]}
    if result is None:
        result = {}

    interrupt_payload = _extract_interrupt_payload(result)
    reply = _interrupt_question(interrupt_payload) if interrupt_payload else _latest_ai_text(result)
    return TurnOutcome(
        user_text=user_text or resume_text or "",
        reply=reply,
        result=result,
        thread_id=str(config["configurable"]["thread_id"]),
        agent_status="interrupted" if interrupt_payload else "completed",
        interrupt_payload=interrupt_payload,
    )


async def _get_pending_interrupt_payload(graph: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = await _get_state_snapshot(graph, config)
    if snapshot is None:
        return None
    interrupts = getattr(snapshot, "interrupts", None) or []
    if interrupts:
        return _coerce_interrupt_payload(interrupts[-1])
    values = getattr(snapshot, "values", None)
    return _extract_interrupt_payload(values if isinstance(values, dict) else None)


def _save_interactive_transcript(transcript: list[tuple[str, str]]) -> None:
    markdown = "\n".join(f"## {speaker}\n{text}\n" for speaker, text in transcript)
    _save_transcript(_transcript_path("interactive_last"), markdown)


async def _resume_pending_interrupts(
    *,
    graph: Any,
    config: dict[str, Any],
    transcript: list[tuple[str, str]],
    current_scenario: str,
    initial_payload: dict[str, Any],
    already_announced: bool = False,
) -> tuple[str, bool]:
    payload = initial_payload
    question = _interrupt_question(payload)
    if question and not already_announced:
        print(f"\nAgent: {question}\n")
        transcript.append(("agent", question))
        _save_interactive_transcript(transcript)
    while True:
        raw_text = _read_text_input("User> ")
        handled, current_scenario, should_reset = await _handle_runtime_command(
            raw_text,
            graph=graph,
            config=config,
            scenario=current_scenario,
        )
        if should_reset:
            return current_scenario, True
        if handled:
            continue
        resume_text = raw_text.strip()
        if not resume_text:
            continue
        transcript.append(("user", resume_text))
        outcome = await _invoke_agent(graph, config, None, resume_text=resume_text)
        _save_persistent_interactive_session(config=config, scenario=current_scenario)
        print(f"\nAgent: {outcome.reply}\n")
        transcript.append(("agent", outcome.reply))
        _save_interactive_transcript(transcript)
        if outcome.agent_status != "interrupted" or not outcome.interrupt_payload:
            return current_scenario, False


def _transcript_path(name: str | None = None) -> Path:
    root = Path("./data/simulator_transcripts")
    root.mkdir(parents=True, exist_ok=True)
    stem = name or f"session_{uuid.uuid4().hex}"
    return root / f"{stem}.md"


def _has_document_evidence(result: dict[str, Any]) -> bool:
    contract = result.get("normalized_final_answer") or {}
    for item in contract.get("items") or []:
        for evidence in item.get("evidence") or []:
            if (
                evidence.get("source") == "document"
                and str(evidence.get("snippet") or "").strip()
                and (evidence.get("file_path") or evidence.get("source_url"))
                and (evidence.get("page") is not None or evidence.get("locator"))
            ):
                return True
    return False


def _tool_usage_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    usage = result.get("turn_tool_usage") or []
    return [entry for entry in usage if isinstance(entry, dict)]


def _used_tools(result: dict[str, Any]) -> set[str]:
    return {
        str(entry.get("tool"))
        for entry in _tool_usage_entries(result)
        if entry.get("tool") and entry.get("status") in {"success", "partial"}
    }


def _has_valid_fact_labels(result: dict[str, Any]) -> bool:
    valid_fact_statuses = {"document", "external_api", "open_source", "not_found"}
    valid_evidence_sources = {"purchase", "document", "open_source", "scoring", "fssp"}
    contract = result.get("normalized_final_answer") or {}
    for item in contract.get("items") or []:
        for evidence in item.get("evidence") or []:
            if evidence.get("source") not in valid_evidence_sources:
                return False
        for fact_status in item.get("fact_statuses") or []:
            if fact_status.get("status") not in valid_fact_statuses:
                return False
    return True


def _has_searchable_index_support(result: dict[str, Any]) -> bool:
    if not result.get("index_id"):
        return False
    prepared_documents = result.get("prepared_documents") or []
    if any(
        isinstance(document, dict)
        and document.get("index_status") == "ready"
        and int(document.get("chunks_count") or 0) > 0
        for document in prepared_documents
    ):
        return True
    last_doc_search = result.get("last_doc_search_result") or {}
    return bool(last_doc_search.get("matches"))


def _item_has_visible_support(item: dict[str, Any]) -> bool:
    valid_evidence_sources = {"purchase", "document", "open_source", "scoring", "fssp"}
    valid_fact_statuses = {"document", "external_api", "open_source", "not_found"}
    has_evidence = False
    has_fact_status = False
    for evidence in item.get("evidence") or []:
        if evidence.get("source") in valid_evidence_sources and str(evidence.get("snippet") or "").strip():
            has_evidence = True
    for fact_status in item.get("fact_statuses") or []:
        if fact_status.get("status") in valid_fact_statuses:
            has_fact_status = True
    return has_evidence and has_fact_status


def _has_visible_support(result: dict[str, Any]) -> bool:
    contract = result.get("normalized_final_answer") or {}
    items = contract.get("items") or []
    return bool(items) and all(_item_has_visible_support(item) for item in items)


def _has_document_fact_labels(result: dict[str, Any]) -> bool:
    contract = result.get("normalized_final_answer") or {}
    items = contract.get("items") or []
    if not items:
        return False
    for item in items:
        evidence = item.get("evidence") or []
        if not any(entry.get("source") == "document" for entry in evidence):
            continue
        statuses = {entry.get("status") for entry in item.get("fact_statuses") or []}
        if "document" not in statuses:
            return False
    return True


def _requires_next_step(result: dict[str, Any]) -> bool:
    normalized = result.get("normalized_final_answer") or {}
    if normalized.get("missing_data"):
        return True
    validation = result.get("turn_validation") or {}
    return bool(validation.get("issues"))


def _validate_outcome(
    *,
    scenario_name: str,
    step_index: int,
    outcome: TurnOutcome,
    expectation: ScenarioExpectation,
    previous: TurnOutcome | None,
) -> None:
    normalized = outcome.result.get("normalized_final_answer") or {}
    errors: list[str] = []
    if expectation.answer_type and normalized.get("answer_type") != expectation.answer_type:
        errors.append(
            f"expected answer_type={expectation.answer_type}, got {normalized.get('answer_type')!r}"
        )

    used_tools = _used_tools(outcome.result)
    for tool_name in expectation.required_tools:
        if tool_name not in used_tools:
            errors.append(f"required tool {tool_name} was not recorded for this turn")

    active_run_id = outcome.result.get("active_run_id")
    if expectation.require_active_run and not active_run_id:
        errors.append("active_run_id is missing")
    if expectation.require_active_run and not outcome.result.get("index_id"):
        errors.append("index_id is missing for a run-scoped scenario")
    if (
        expectation.require_searchable_index
        and not _has_searchable_index_support(outcome.result)
    ):
        errors.append("searchable prepared index support is missing")

    if previous is not None and outcome.thread_id != previous.thread_id:
        errors.append("thread_id changed inside a scripted multi-turn scenario")

    if expectation.reuse_active_run_from_previous:
        previous_run_id = previous.result.get("active_run_id") if previous else None
        if not previous_run_id or active_run_id != previous_run_id:
            errors.append("active_run_id was not reused from the previous turn")

    if (
        expectation.require_document_evidence
        and not _has_document_evidence(outcome.result)
    ):
        errors.append("document evidence is missing from the final normalized answer")
    if (
        expectation.require_document_fact_labels
        and not _has_document_fact_labels(outcome.result)
    ):
        errors.append("document-backed answer is missing explicit document fact-status labels")
    if (
        expectation.require_visible_support
        and not _has_visible_support(outcome.result)
    ):
        errors.append("visible evidence or fact-status support is missing from the final normalized answer")

    if not _has_valid_fact_labels(outcome.result):
        errors.append("fact-source labeling is incomplete or invalid")

    if _requires_next_step(outcome.result) and not normalized.get("recommended_next_step"):
        errors.append("partial or failed turn is missing recommended_next_step")

    if errors:
        joined = "; ".join(errors)
        raise AssertionError(
            f"Scenario '{scenario_name}' step {step_index + 1} failed validation: {joined}"
        )


def _scripted_transcript_markdown(outcomes: list[TurnOutcome]) -> str:
    lines: list[str] = []
    for index, outcome in enumerate(outcomes, start=1):
        normalized = outcome.result.get("normalized_final_answer") or {}
        lines.append(f"## Step {index}")
        lines.append(f"Thread ID: `{outcome.thread_id}`")
        lines.append(f"User: {outcome.user_text}")
        lines.append(f"Answer Type: `{normalized.get('answer_type', '')}`")
        if outcome.result.get("active_run_id"):
            lines.append(f"Active Run: `{outcome.result.get('active_run_id')}`")
        if outcome.result.get("index_id"):
            lines.append(f"Index ID: `{outcome.result.get('index_id')}`")
        usage_entries = _tool_usage_entries(outcome.result)
        if usage_entries:
            tool_parts = [
                f"`{entry.get('tool')}`:{entry.get('status')}"
                for entry in usage_entries
                if entry.get("tool")
            ]
            lines.append("Tools: " + ", ".join(tool_parts))
        lines.append("")
        lines.append("### Agent")
        lines.append(outcome.reply or "(empty reply)")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _save_transcript(path: Path, markdown: str) -> None:
    path.write_text(markdown, encoding="utf-8")


async def run_scripted_scenario(*, provider: ModelType, scenario: str) -> Path:
    async with _persistent_checkpoint_saver() as checkpoint_saver:
        await checkpoint_saver.setup()
        graph = initialize_agent(
            provider=provider,
            streaming=False,
            checkpoint_saver=checkpoint_saver,
        )
        config = _new_config()
        outcomes: list[TurnOutcome] = []
        previous: TurnOutcome | None = None
        for step_index, step in enumerate(SCENARIOS[scenario]):
            outcome = await _invoke_agent(graph, config, step.user_text)
            _validate_outcome(
                scenario_name=scenario,
                step_index=step_index,
                outcome=outcome,
                expectation=step.expectation,
                previous=previous,
            )
            outcomes.append(outcome)
            previous = outcome
        path = _transcript_path(scenario)
        _save_transcript(path, _scripted_transcript_markdown(outcomes))
        return path


async def run_interactive(*, provider: ModelType, scenario: str) -> None:
    async with _persistent_checkpoint_saver() as checkpoint_saver:
        await checkpoint_saver.setup()
        graph = initialize_agent(
            provider=provider,
            streaming=False,
            checkpoint_saver=checkpoint_saver,
        )
        config, current_scenario, restored_session = _load_persistent_interactive_session(default_scenario=scenario)
        _save_persistent_interactive_session(config=config, scenario=current_scenario)
        transcript: list[tuple[str, str]] = []
        print("\nSales lead agent simulator")
        print(f"Scenario: {current_scenario}")
        print(f"Thread ID: {config['configurable']['thread_id']}")
        if restored_session:
            print("[Simulator] restored previous dialog session.")
        print(HELP_TEXT)

        pending_interrupt = await _get_pending_interrupt_payload(graph, config)
        if pending_interrupt:
            print("[Simulator] restored pending confirmation for this dialog.")
            current_scenario, should_reset = await _resume_pending_interrupts(
                graph=graph,
                config=config,
                transcript=transcript,
                current_scenario=current_scenario,
                initial_payload=pending_interrupt,
                already_announced=False,
            )
            if should_reset:
                config = _new_config()
                _save_persistent_interactive_session(config=config, scenario=current_scenario)
                transcript = []
                print(f"\n[Simulator] new dialog session started: {config['configurable']['thread_id']}\n")

        while True:
            raw_text = _read_text_input("User> ", allow_multiline=True)
            handled, current_scenario, should_reset = await _handle_runtime_command(
                raw_text,
                graph=graph,
                config=config,
                scenario=current_scenario,
            )
            if should_reset:
                config = _new_config()
                _save_persistent_interactive_session(config=config, scenario=current_scenario)
                transcript = []
                print(f"\n[Simulator] new dialog session started: {config['configurable']['thread_id']}\n")
                continue
            if handled:
                _save_persistent_interactive_session(config=config, scenario=current_scenario)
                continue
            user_text = raw_text.strip()
            if not user_text:
                continue
            transcript.append(("user", user_text))
            outcome = await _invoke_agent(graph, config, user_text)
            _save_persistent_interactive_session(config=config, scenario=current_scenario)
            print(f"\nAgent: {outcome.reply}\n")
            transcript.append(("agent", outcome.reply))
            _save_interactive_transcript(transcript)
            if outcome.agent_status == "interrupted" and outcome.interrupt_payload:
                current_scenario, should_reset = await _resume_pending_interrupts(
                    graph=graph,
                    config=config,
                    transcript=transcript,
                    current_scenario=current_scenario,
                    initial_payload=outcome.interrupt_payload,
                    already_announced=True,
                )
                if should_reset:
                    config = _new_config()
                    _save_persistent_interactive_session(config=config, scenario=current_scenario)
                    transcript = []
                    print(f"\n[Simulator] new dialog session started: {config['configurable']['thread_id']}\n")


async def run_all_scenarios(*, provider: ModelType) -> list[Path]:
    paths = []
    for scenario in SCENARIOS:
        paths.append(await run_scripted_scenario(provider=provider, scenario=scenario))
    return paths


def main() -> None:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Simulator for sales_lead_agent.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS.keys()))
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--interactive", default=True, action="store_true")
    args = parser.parse_args()

    provider = _parse_provider(args.provider)
    try:
        if args.all_scenarios:
            paths = asyncio.run(run_all_scenarios(provider=provider))
            for path in paths:
                print(path)
            return
        scenario = args.scenario or "procurement_search"
        if args.interactive:
            asyncio.run(run_interactive(provider=provider, scenario=scenario))
            return
        path = asyncio.run(run_scripted_scenario(provider=provider, scenario=scenario))
        print(path)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
