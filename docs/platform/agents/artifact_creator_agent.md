# `artifact_creator_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

Interactive drafting agent for user-defined artifacts. It captures a prompt, generates an artifact candidate, asks for confirmation, and publishes the final result as a downloadable artifact bundle.

## Entry module and initialization

- Implementation: `agents/artifact_creator_agent/agent.py`
- Contract: exposes `initialize_agent(provider, use_platform_store, locale, checkpoint_saver, tools=None, system_prompt=None, guardrails_enabled=False, guardrails_locale="ru-RU", guardrail_scanners_enabled=None, guardrail_scanner_failure_policy="fail_closed", guardrail_banned_topics=None, guardrail_composite_input_scanners=None, guardrail_composite_recent_message_limit=20, guardrail_palimpsest_run_entities=None, guardrail_palimpsest_entity_table=None, guardrail_palimpsest_typed_placeholders=None, guardrail_palimpsest_options=None, guardrail_palimpsest_session_options=None)`
- Registry variant: `artifact_creator_agent`

## State graph / phases / routing

- `greetings`
- `set_prompt`
- `cleanup`
- `run`
- `confirm`
- `final_print`
- `ready`

The `run` node uses a LangChain agent with a commit tool. A second confirmation agent classifies the user reply and either loops back for revisions or finalizes the artifact.

## Inputs, context, and attachments

- Primary input is the user-defined system prompt and subsequent revision feedback.
- A fixed `system_prompt` can be injected through `initialize_agent(...)`; when provided there, runtime context prompt settings are ignored.
- No attachment support is declared in the registry.
- Uses conversation memory or the injected checkpoint saver.

## Tools and integrations

- `commit_artifact_final_text`
- optional extra tools passed through `initialize_agent(..., tools=[...])`
- optional fixed prompt passed through `initialize_agent(..., system_prompt="...")`
- `agents/store_artifacts.py`
- `platform_guardrails.logging.RedactingJSONFileTracer`
- Phase 1 privacy foundation for guarded runs; see [Guardrails Implementation Status](../../guardrails/implementation_status.md)
- Phase 2 scanner enforcement for guarded runs; see [Scanner Enforcement Vulnerability Catalog](../../guardrails/scanner_enforcement_vulnerabilities.md)
- The prompt-setup node is wrapped with the common guardrail node wrapper when guardrails are enabled, so user-provided system prompts are scanned before being stored.
- optional Langfuse callbacks when configured; this is allowed regardless of `guardrails_enabled` because the configured Langfuse endpoint is treated as trusted platform observability
- Palimpsest privacy can be configured per registry variant. `guardrail_palimpsest_entity_table` defines the agent-specific entity table, `guardrail_palimpsest_typed_placeholders=True` requests typed placeholder replacement from compatible Palimpsest versions, and `guardrail_palimpsest_options` / `guardrail_palimpsest_session_options` pass through library-specific constructor/session options.

## Outputs and persistence

- Produces `artifact_final_text` in agent state.
- Uploads the final artifact and returns a completion message with a link.
- Conversation messages are persisted by `bot_service`; artifact files are stored outside the SQL conversation history.

## Special behavior

- Confirmation is model-driven rather than a fixed yes/no string match.
- Final artifact publishing is separate from conversational persistence.
