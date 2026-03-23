from __future__ import annotations

import argparse
import asyncio
import locale
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import termios
except ImportError:  # pragma: no cover - Windows
    termios = None

from langchain_core.messages import AIMessage, HumanMessage

from agents.sales_lead_agent.agent import initialize_agent
from agents.sales_lead_agent.settings import get_settings
from agents.utils import ModelType, extract_text


DEFAULT_PROVIDER = ModelType.GPT.value
HELP_TEXT = """
Commands:
  /scenario <name>  switch scripted scenario
  /reset            start a fresh dialog session
  /help             show commands
  /exit             leave the simulator

Use /multiline ... /end for multi-line input.
""".strip()
_MULTILINE_START = "/multiline"
_MULTILINE_END = "/end"
_INPUT_FALLBACK_ENCODINGS = ("utf-8", "cp1251", "cp866", "latin-1")


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


SCENARIOS: dict[str, list[ScenarioStep]] = {
    "procurement_search": [
        ScenarioStep(
            "Find new relevant transport insurance procurements.",
            ScenarioExpectation(
                answer_type="lead_list",
                required_tools=("purchase_search_tool",),
                require_active_run=True,
                require_searchable_index=True,
                require_visible_support=True,
            ),
        ),
    ],
    "procurement_analysis": [
        ScenarioStep(
            "Analyze one transport insurance procurement and summarize the main risks.",
            ScenarioExpectation(
                answer_type="lead_card",
                required_tools=("purchase_search_tool", "doc_search_tool"),
                require_active_run=True,
                require_searchable_index=True,
                require_document_evidence=True,
                require_document_fact_labels=True,
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
            ScenarioExpectation(
                answer_type="lead_card",
                required_tools=("doc_search_tool",),
                require_document_evidence=True,
                require_document_fact_labels=True,
            ),
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
                answer_type="lead_list",
                required_tools=("purchase_search_tool",),
                require_active_run=True,
                require_searchable_index=True,
                require_visible_support=True,
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
                answer_type="lead_card",
                required_tools=("purchase_search_tool", "doc_search_tool"),
                require_active_run=True,
                require_searchable_index=True,
                require_document_evidence=True,
                require_document_fact_labels=True,
            ),
        ),
        ScenarioStep(
            "Show the source snippet for the main risk.",
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
                answer_type="lead_list",
                required_tools=("purchase_search_tool",),
                require_active_run=True,
                require_searchable_index=True,
                require_visible_support=True,
            ),
        ),
        ScenarioStep(
            "Find new relevant transport insurance procurements.",
            ScenarioExpectation(
                answer_type="lead_list",
                required_tools=("purchase_search_tool",),
                require_active_run=True,
                require_searchable_index=True,
                reuse_active_run_from_previous=True,
                require_visible_support=True,
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


def _latest_ai_text(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages") or []):
        if isinstance(message, AIMessage):
            return extract_text(message)
    return ""


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


async def _invoke_agent(graph: Any, config: dict[str, Any], user_text: str | None) -> TurnOutcome:
    payload = {} if user_text is None else {"messages": [HumanMessage(content=user_text)]}
    result = await graph.ainvoke(payload, config=config)
    return TurnOutcome(
        user_text=user_text or "",
        reply=_latest_ai_text(result),
        result=result,
        thread_id=str(config["configurable"]["thread_id"]),
    )


def _transcript_path(name: str | None = None) -> Path:
    root = get_settings().work_root.parent / "simulator_transcripts"
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
    if expectation.require_searchable_index and not _has_searchable_index_support(outcome.result):
        errors.append("searchable prepared index support is missing")

    if previous is not None and outcome.thread_id != previous.thread_id:
        errors.append("thread_id changed inside a scripted multi-turn scenario")

    if expectation.reuse_active_run_from_previous:
        previous_run_id = previous.result.get("active_run_id") if previous else None
        if not previous_run_id or active_run_id != previous_run_id:
            errors.append("active_run_id was not reused from the previous turn")

    if expectation.require_document_evidence and not _has_document_evidence(outcome.result):
        errors.append("document evidence is missing from the final normalized answer")
    if expectation.require_document_fact_labels and not _has_document_fact_labels(outcome.result):
        errors.append("document-backed answer is missing explicit document fact-status labels")
    if expectation.require_visible_support and not _has_visible_support(outcome.result):
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
    graph = initialize_agent(provider=provider, streaming=False)
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
    graph = initialize_agent(provider=provider, streaming=False)
    current_scenario = scenario
    config = _new_config()
    transcript: list[tuple[str, str]] = []
    print("\nSales lead agent simulator")
    print(f"Scenario: {current_scenario}")
    print(HELP_TEXT)

    while True:
        raw_text = _read_text_input("User> ", allow_multiline=True)
        handled, current_scenario, should_reset = _handle_command(raw_text, scenario=current_scenario)
        if should_reset:
            config = _new_config()
            transcript = []
            print("\n[Simulator] new dialog session started.\n")
            continue
        if handled:
            continue
        user_text = raw_text.strip()
        if not user_text:
            continue
        transcript.append(("user", user_text))
        outcome = await _invoke_agent(graph, config, user_text)
        print(f"\nAgent: {outcome.reply}\n")
        transcript.append(("agent", outcome.reply))
        markdown = "\n".join(f"## {speaker}\n{text}\n" for speaker, text in transcript)
        _save_transcript(_transcript_path("interactive_last"), markdown)


async def run_all_scenarios(*, provider: ModelType) -> list[Path]:
    paths = []
    for scenario in SCENARIOS:
        paths.append(await run_scripted_scenario(provider=provider, scenario=scenario))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulator for sales_lead_agent.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS.keys()))
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--interactive", action="store_true")
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
