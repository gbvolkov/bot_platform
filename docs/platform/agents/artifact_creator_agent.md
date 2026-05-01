# `artifact_creator_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

Interactive drafting agent for user-defined artifacts. It captures a prompt, generates an artifact candidate, asks for confirmation, and publishes the final result as a downloadable artifact bundle.

## Entry module and initialization

- Implementation: `agents/artifact_creator_agent/agent.py`
- Contract: exposes `initialize_agent(provider, use_platform_store, locale, checkpoint_saver, tools=None, system_prompt=None)`
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
- optional Langfuse callbacks when configured; this is allowed regardless of `guardrails_enabled` because the configured Langfuse endpoint is treated as trusted platform observability

## Outputs and persistence

- Produces `artifact_final_text` in agent state.
- Uploads the final artifact and returns a completion message with a link.
- Conversation messages are persisted by `bot_service`; artifact files are stored outside the SQL conversation history.

## Special behavior

- Confirmation is model-driven rather than a fixed yes/no string match.
- Final artifact publishing is separate from conversational persistence.
