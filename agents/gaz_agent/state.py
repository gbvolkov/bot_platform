from __future__ import annotations

from typing import Annotated, Any, Dict, List, NotRequired, Optional

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _latest(a, b):
    return b if b is not None else a


def _merge_list(a: Optional[List[Any]], b: Optional[List[Any]]) -> List[Any]:
    if b is None:
        return list(a or [])
    return list(b)


def _merge_dict(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(a or {})
    merged.update(b or {})
    return merged


class GazAgentState(AgentState[Dict[str, Any]]):
    locale: NotRequired[Annotated[str, _latest]]
    messages: Annotated[List[BaseMessage], add_messages]

    stage: NotRequired[Annotated[str, _latest]]
    greeted: NotRequired[Annotated[bool, _latest]]
    reset_done: NotRequired[Annotated[bool, _latest]]
    last_user_text: NotRequired[Annotated[Optional[str], _latest]]

    slots: NotRequired[Annotated[Dict[str, Any], _merge_dict]]
    missing_slots: NotRequired[Annotated[List[str], _merge_list]]
    intent_flags: NotRequired[Annotated[Dict[str, Any], _merge_dict]]
    problem_summary: NotRequired[Annotated[str, _latest]]

    current_client_intent: NotRequired[Annotated[Optional[str], _latest]]
    planned_answer_depth: NotRequired[Annotated[Optional[str], _latest]]
    answer_depth: NotRequired[Annotated[Optional[str], _latest]]
    customer_temperature: NotRequired[Annotated[Optional[str], _latest]]
    clarification_allowed: NotRequired[Annotated[bool, _latest]]
    validator_guidance: NotRequired[Annotated[str, _latest]]
    llm_retry_instruction: NotRequired[Annotated[str, _latest]]
    provisional_recommendations: NotRequired[Annotated[List[str], _merge_list]]
    hitl_eligible: NotRequired[Annotated[bool, _latest]]
    hitl_blocked_by_temperature: NotRequired[Annotated[bool, _latest]]
    hitl_blocked_by_first_turn_budget: NotRequired[Annotated[bool, _latest]]
    hitl_blocked_by_missing_prior_search: NotRequired[Annotated[bool, _latest]]
    hitl_trigger_kind: NotRequired[Annotated[Optional[str], _latest]]

    active_branch: NotRequired[Annotated[Optional[str], _latest]]
    branch_conflict: NotRequired[Annotated[List[str], _merge_list]]
    docs_status: NotRequired[Annotated[Dict[str, Any], _merge_dict]]
    sales_context_baseline: NotRequired[Annotated[Dict[str, Any], _merge_dict]]
    research_layer: NotRequired[Annotated[Optional[str], _latest]]

    search_query: NotRequired[Annotated[Optional[str], _latest]]
    material_candidates: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]
    material_reads: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]
    allowed_material_ids: NotRequired[Annotated[List[str], _merge_list]]
    sales_digests: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]
    comparison_digests: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]
    product_snapshots: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]
    composite_tool_traces: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]
    shortlist: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]
    followup_pack: NotRequired[Annotated[Dict[str, Any], _merge_dict]]

    research_status: NotRequired[Annotated[Dict[str, Any], _merge_dict]]
    needs_hitl_wait_confirmation: NotRequired[Annotated[bool, _latest]]
    hitl_reason: NotRequired[Annotated[Optional[str], _latest]]
    research_wait_approved: NotRequired[Annotated[bool, _latest]]
    research_wait_rejected: NotRequired[Annotated[bool, _latest]]
    read_attempts_by_candidate: NotRequired[Annotated[Dict[str, Any], _merge_dict]]
    read_focus_keys_this_turn: NotRequired[Annotated[List[str], _merge_list]]
    search_keys_this_turn: NotRequired[Annotated[List[str], _merge_list]]
    tool_limit_hits: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]
    sales_loop_guard_reason: NotRequired[Annotated[Optional[str], _latest]]

    allowed_tool_names: NotRequired[Annotated[List[str], _merge_list]]
    tool_calls_this_turn: NotRequired[Annotated[List[str], _merge_list]]
    draft_answer: NotRequired[Annotated[Optional[str], _latest]]
    validation_feedback: NotRequired[Annotated[Dict[str, Any], _merge_dict]]
    runtime_warnings: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]

    trace: NotRequired[Annotated[List[Dict[str, Any]], _merge_list]]
