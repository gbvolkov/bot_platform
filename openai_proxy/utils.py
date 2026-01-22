from __future__ import annotations

from typing import List, Tuple

from .schemas import ChatMessage


def build_prompt(
    messages: List[ChatMessage],
    *,
    default_user_prompt: str | None = None,
) -> Tuple[str, bool]:
    if not messages:
        return "", False

    system_chunks: List[str] = []
    conversation_chunks: List[str] = []
    latest_user_text: str | None = None
    default_used = False

    for message in messages:
        if message.role == "system":
            system_chunks.append(message.content)
        elif message.role == "assistant":
            conversation_chunks.append(f"Assistant: {message.content}")
        elif message.role == "user":
            content = message.content or ""
            if not content.strip() and default_user_prompt:
                content = default_user_prompt
                default_used = True
            latest_user_text = content
            conversation_chunks.append(f"User: {content}")

    if latest_user_text is None:
        raise ValueError("Chat request must include at least one user message.")

    prompt_sections: List[str] = []
    #if system_chunks:
    #    prompt_sections.append("\n".join(system_chunks))
    #if len(conversation_chunks) > 1:
    #    prompt_sections.append("Conversation history:\n" + "\n".join(conversation_chunks[:-1]))
    prompt_sections.append(latest_user_text)

    prompt = "\n\n".join(section for section in prompt_sections if section).strip()
    if not prompt and default_user_prompt:
        prompt = default_user_prompt
        default_used = default_used or bool(default_user_prompt)

    return prompt, default_used
