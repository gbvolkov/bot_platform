from __future__ import annotations


_DELEGATE_TOOL_DESCRIPTION = """Consult a stateful team member to handle expert work that may continue across multiple user turns.

Available agent types and the tools they have access to:
{available_agents}

When using the `delegate` tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Use `delegate` for specialists who should keep their own local context and state across calls on the same conversation thread.
2. You may call the same team member again on later user turns. Provide the next natural-language message or clarification they should see now.
3. Do not call the same stateful agent multiple times in parallel within a single turn.
4. Use `delegate` when you want an ongoing working conversation with a specialist, not a one-shot isolated task.
5. The delegated agent returns a message back to you. Summarize or integrate it for the user as needed."""


_DELEGATE_SYSTEM_PROMPT = """## `delegate` (stateful team members)

You have access to a `delegate` tool to consult stateful team members. These agents keep their own local state and history across calls on the same conversation thread.

When to use `delegate`:
- When you need an ongoing working dialogue with a specialist
- When the same expert may need follow-up clarifications or additional instructions on later user turns
- When the specialist should keep their own local context instead of receiving a full re-brief every time

How to use `delegate`:
1. Pick the right stateful team member with `subagent_type`
2. Send a natural-language message describing what the team member should do or what new clarification they should consider
3. If the same specialist needs follow-up later, call `delegate` again for that same agent
4. Do not call the same stateful agent multiple times in parallel within a single turn

When not to use `delegate`:
- If the work is best handled as an isolated one-shot task; use `task` for that
- If a direct tool call is simpler"""


def build_delegate_tool_description(*, tool_name: str = "delegate") -> str:
    return _DELEGATE_TOOL_DESCRIPTION.replace("`delegate`", f"`{tool_name}`")


def build_delegate_system_prompt(*, tool_name: str = "delegate") -> str:
    return _DELEGATE_SYSTEM_PROMPT.replace("`delegate`", f"`{tool_name}`")
