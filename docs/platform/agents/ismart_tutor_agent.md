# `ismart_tutor_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

Educational tutor agent that gives hints rather than full solutions. It collects a student profile, asks for missing details, and then generates a tailored hint based on text and optional images.

## Entry module and initialization

- Implementation: `agents/ismart_tutor_agent/agent.py`
- Contract: exposes `initialize_agent(provider=ModelType.GPT_PERS, use_platform_store, checkpoint_saver)`
- Registry variant: `ismart_tutor_agent`

## State graph / phases / routing

- `init`
- `collect_person_info`
- `check_person_info_initial`
- `extract_person_info`
- `check_person_info_after_extraction`
- `ask_person_info` or `ask_task` or `generate_hint`
- `format`

## Inputs, context, and attachments

- Registry declares `supported_content_types=[images]` and `allow_raw_attachments=true`.
- Accepts task text plus optional image attachments.
- Reads and stores a person profile: name, age, school year, and nosology type.
- Supports reset-style cleanup of accumulated profile state.

## Tools and integrations

- Structured extraction agent for person info
- Structured hint-generation agent
- `data/nosologies.json` for allowed profile taxonomy and extra instructions
- JSON tracing and optional Langfuse

## Outputs and persistence

- Returns either:
  - a prompt asking for missing profile/task information, or
  - a final hint message
- Keeps `person_profile`, `last_user_text`, and hint state in the checkpoint.

## Special behavior

- The graph is profile-gated: it deliberately asks for missing student context before it will generate a hint.
