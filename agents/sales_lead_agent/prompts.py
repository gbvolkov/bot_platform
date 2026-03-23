from __future__ import annotations

import json
from typing import Any


BASE_SYSTEM_PROMPT = """
You are `sales_lead_agent`, a pre-sales procurement and counterparty qualification agent.

Core behavior:
- Work in a free tool-calling loop.
- Keep conversational context between turns and reuse the active run when the user asks follow-up questions.
- Use only verified facts from tool outputs and indexed documents.
- Do not invent missing facts.
- Be concise and business-like in the final answer.

Ordering guard:
1. Acquisition and source preparation.
2. Document search and fact extraction.
3. Enrichment by external APIs after a normalized INN exists.
4. Final answer.

Tool policy:
- `purchase_search_tool` is the procurement acquisition path and already prepares documents.
- `open_source_fetch_tool` is a supplement for lead cards and company checks.
- `doc_search_tool` is mandatory for fact lookup inside prepared documents.
- Do not use filesystem or execution tools unless explicitly required for repository-internal troubleshooting; the domain answer must rely on the 5 domain tools.
- For `procurement_search`, if `search_url` is present or `search_filters.query_text` is present, you must call `purchase_search_tool` in the current turn before finalizing.
- For `procurement_search`, optional refinements such as region, price range, publication window, law, customer hints, or competitor hints must not block the first acquisition attempt when a search query already exists.
- Do not end a `procurement_search` turn with only clarification text if a best-effort `purchase_search_tool` call can already be made from the current context.
- For `procurement_analysis`, if the user did not provide a direct procurement URL or registry number but the task understanding contains acquisition-ready search criteria, you must call `purchase_search_tool` first to acquire one relevant procurement before document analysis.
- Do not end a `procurement_analysis` turn with only generic missing-data text if a best-effort `purchase_search_tool` call can already be made from the current context.

Answer policy:
- Output must fit the structured contract.
- Separate confirmed document facts, external API facts, open-source facts, and not-found facts.
- If data is insufficient, say what is missing and suggest the next step in the same dialog.
""".strip()


def _compact_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return str(payload)


def build_system_prompt(state: dict[str, Any] | None = None) -> str:
    state = state or {}
    sections = [BASE_SYSTEM_PROMPT]
    understanding = state.get("task_understanding") or {}
    assessment = state.get("assessment") or {}
    turn_validation = state.get("turn_validation") or {}
    tool_requirements = state.get("turn_tool_requirements") or {}
    tool_usage = state.get("turn_tool_usage") or []
    active_run_id = state.get("active_run_id")
    index_id = state.get("index_id")

    runtime_notes: list[str] = []
    if active_run_id:
        runtime_notes.append(f"Active run: {active_run_id}")
    if index_id:
        runtime_notes.append(f"Active index: {index_id}")
    if understanding:
        runtime_notes.append("Task understanding:\n" + _compact_json(understanding))
    if assessment:
        runtime_notes.append("Latest semantic assessment:\n" + _compact_json(assessment))
    if turn_validation and (
        turn_validation.get("status") != "clean" or turn_validation.get("issues")
    ):
        runtime_notes.append("Current validation status:\n" + _compact_json(turn_validation))
    if tool_requirements:
        runtime_notes.append("Current turn tool requirements:\n" + _compact_json(tool_requirements))
    if tool_usage:
        runtime_notes.append("Current turn tool usage:\n" + _compact_json(tool_usage))
    if (
        tool_requirements.get("purchase_search_required")
        and not tool_usage
        and (
            understanding.get("search_url")
            or ((understanding.get("search_filters") or {}).get("query_text"))
        )
    ):
        runtime_notes.append(
            "Action now: call purchase_search_tool immediately using the available search_url or search_filters."
        )
        runtime_notes.append(
            "Non-blocking refinements for this first procurement attempt: region, price range, time window, procurement law, customer hints."
        )
    procurement_hits = state.get("procurement_hits") or []
    if procurement_hits:
        runtime_notes.append(
            f"Relevant procurement hits in state: {len(procurement_hits)}"
        )
    unclassified_hits = state.get("unclassified_procurement_hits") or []
    if unclassified_hits:
        runtime_notes.append(f"Unclassified procurement hits in state: {len(unclassified_hits)}")
    open_source_hits = state.get("open_source_hits") or []
    if open_source_hits:
        runtime_notes.append(f"Open-source bundles in state: {len(open_source_hits)}")
    normalized_inns = state.get("normalized_inns") or []
    if normalized_inns:
        runtime_notes.append("Known INNs: " + ", ".join(normalized_inns[:10]))
    if runtime_notes:
        sections.append("Current runtime context:\n" + "\n".join(f"- {line}" for line in runtime_notes))
    return "\n\n".join(sections)
