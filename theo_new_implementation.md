# new_theodor_agent implementation plan

Date: 2026-01-30
Owner: Codex
Status: Draft plan only (no code changes yet)

## Goals
- Create a new agent `agents/new_theodor_agent` that follows the simplified techniques in `agents/new_ideator_agent`.
- Remove structured-output dependencies for Theodor flow; store and "fix" artifacts using text-only tool calls.
- Keep artifact state persistence and final report generation, but avoid JSON/TypedDict/Pydantic response formats.
- Focus most changes inside a new `choice_agent` (text-based) while keeping the overall Theodor pipeline behavior.

## Non-goals
- Do not modify existing `agents/theodor_agent` or `agents/ideator_agent`.
- Do not change the shared artifact definitions unless required for compatibility.
- Do not introduce new external dependencies.

## High-level design
- New folder: `agents/new_theodor_agent/`.
- Use a simplified LangGraph flow like `new_ideator_agent`: small, explicit nodes, with SummarizationMiddleware preserving artifact text.
- Replace structured output parsing with tool calls that persist plain text (options, selections, final artifact text, final docset).
- Reuse `agents/theodor_agent/artifacts_defs.py` for artifact metadata and `store_artifacts.py` to generate the final report.

## File layout (new)
- `agents/new_theodor_agent/agent.py`
  - Orchestrates the per-artifact loop (progress banner -> choice_agent -> optional cleanup -> confirmed banner), but uses a text-based choice agent.
  - Uses SummarizationMiddleware with a summary prompt that explicitly preserves artifact text and selected options.
- `agents/new_theodor_agent/choice_agent.py`
  - Core changes live here. Implements text-only option generation, selection, and final artifact generation.
  - No response_format and no structured outputs.
- `agents/new_theodor_agent/tools.py`
  - Tool functions to persist artifact data in state (text-only).
- `agents/new_theodor_agent/state.py`
  - Minimal state schema: messages, user_prompt, current_artifact_id/state, artifacts dict with text fields.
- `agents/new_theodor_agent/prompts.py`
  - System + stage prompts; explicit tool-call instructions for persisting text artifacts.

## State shape (text-only)
- `artifacts: Dict[int, Dict[str, Any]]` with fields:
  - `artifact_definition`: copied from `artifacts_defs`.
  - `artifact_options_text`: string (raw options list formatted for the user).
  - `selected_option_text`: string (the chosen option text or label).
  - `artifact_final_text`: string (final artifact body + criteria assessment).
  - `artifact_summary`: optional summary string for pruning history.
- `current_artifact_id: int`
- `current_artifact_state`: enum-like string (INIT, OPTIONS_GENERATED, OPTION_SELECTED, ARTIFACT_GENERATED, ARTIFACT_CONFIRMED)
- `messages`, `user_prompt`, plus any existing required fields.

## Tools (text persistence)
Add text-only tools in `agents/new_theodor_agent/tools.py`:
- `commit_artifact_options(artifact_id: int, options_text: str)`
- `commit_artifact_selection(artifact_id: int, selection_text: str)`
- `commit_artifact_final_text(artifact_id: int, final_text: str)`
- `commit_final_docset(final_doc_set: str)` (optional, if we want a single final package text)

Each tool returns a `Command(update={...})` with the text stored in state and a `ToolMessage("Success")`.

## Prompting changes
- Replace JSON contracts with plain-language instructions plus explicit tool-call guidance, similar to new_ideator_agent:
  - After generating options, ALWAYS call `commit_artifact_options` with the full options text.
  - After user selection, include the selected option in `commit_artifact_selection`.
  - After generating the final artifact, ALWAYS call `commit_artifact_final_text` with the exact final text (no extra commentary).
- Add a summarization prompt (like `SUMMARY_PROMPT`) that mandates preserving:
  - All artifact final texts
  - Selected options
  - Current artifact id/state
- Keep Theodor rules (only work on current artifact, no other artifacts) but remove JSON formatting directives.

## Flow changes (choice_agent)
Major changes happen here.

### Existing flow (today)
- Options -> structured output -> selection -> structured output -> confirmation.

### New flow (text-only)
1. Init
   - Set `user_prompt`, `current_artifact_id`, `current_artifact_state=INIT`.
2. Generate options (LLM)
   - Prompt includes artifact context, criteria, and constraints.
   - LLM produces options text and calls `commit_artifact_options`.
   - Agent sends options to user with a single-question prompt (choose A/B/C or request edits).
   - State: `OPTIONS_GENERATED`.
3. Select option (local parsing)
   - Parse user response to detect confirm/edit or a selection label.
   - Store selection text via `commit_artifact_selection` (best-effort: label or extracted snippet).
   - State: `OPTION_SELECTED`.
4. Generate final artifact (LLM)
   - Prompt includes selected option text and constraints.
   - LLM produces final text and calls `commit_artifact_final_text`.
   - Agent sends final text + confirmation question.
   - State: `ARTIFACT_GENERATED`.
5. Confirmation
   - If user confirms: mark `ARTIFACT_CONFIRMED`.
   - If user requests change: loop back to "Generate final artifact" (do NOT regenerate options unless explicitly requested).

## Flow changes (top-level agent)
- Keep current per-artifact loop in `theodor_agent/agent.py`, but point each step to the new text-based `choice_agent`.
- Replace context pruning with SummarizationMiddleware (like new_ideator_agent) OR keep pruning but ensure it preserves `artifact_final_text` and `artifact_options_text`.
- The final output node will continue calling `store_artifacts` using the new text fields.
  - If `store_artifacts` expects structured fields, adapt it in the new agent layer (mapping text fields to the expected output format) without changing the shared module.

## Compatibility notes
- Reuse `ARTIFACTS` definitions from `agents/theodor_agent/artifacts_defs.py`.
- Reuse `store_artifacts.py` but may need an adapter if it expects structured data.
- Keep locale handling consistent (RU/EN) but avoid JSON format instructions.

## Testing plan
- Manual run using `simulate_theodor_full_flow.py` style flow (optional local script copy for new agent).
- Verify for a short run:
  - Options text is stored in state via tool call.
  - Final artifact text is stored and appears in `store_artifacts` output.
  - Summarization does not drop artifact texts.

## Implementation steps (detailed)
1. Create `agents/new_theodor_agent/` with `agent.py`, `choice_agent.py`, `state.py`, `tools.py`, `prompts.py`.
2. Implement `state.py` with a minimal `ArtifactAgentState` that mirrors needed fields (messages, user_prompt, artifacts dict, current_artifact_id/state).
3. Implement `tools.py` with text-only commit tools.
4. Port key prompt content from `agents/theodor_agent/prompts` but remove JSON/structured formatting and add tool-call instructions.
5. Build `choice_agent.py`:
   - Implement init, generate_options, select_option, generate_final, confirm nodes.
   - Use `SummarizationMiddleware` with a summary prompt that preserves artifact text.
   - Use LLM tools list: commit_* tools (no structured response_format).
6. Build `agent.py`:
   - Create graph that loops artifacts similarly to current Theodor agent.
   - Use new `choice_agent` per artifact.
   - Keep progress/confirmed banners (optional) or simplified equivalents.
7. Add adapter in new agent layer if `store_artifacts` requires structured artifact fields.
8. Sanity run with a short conversation to confirm text-based persistence and final output.

## Open questions
- Should the new choice agent allow regenerating options on explicit user request, or only regenerating final text?
- Do we want to keep the banner UI (progress/confirmed) or simplify like new_ideator_agent?
- Does `store_artifacts` accept plain text fields or will we add a small adapter in the new agent layer?

