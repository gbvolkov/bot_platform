from __future__ import annotations

from typing import List

from .schemas import ChatMessage


def build_prompt(messages: List[ChatMessage]) -> str:
    if not messages:
        return ""

    system_chunks: List[str] = []
    conversation_chunks: List[str] = []
    latest_user_text: str | None = None

    for message in messages:
        if message.role == "system":
            system_chunks.append(message.content)
        elif message.role == "assistant":
            conversation_chunks.append(f"Assistant: {message.content}")
        elif message.role == "user":
            latest_user_text = message.content
            conversation_chunks.append(f"User: {message.content}")

    if latest_user_text is None:
        raise ValueError("Chat request must include at least one user message.")

    prompt_sections: List[str] = []
    if system_chunks:
        prompt_sections.append("\n".join(system_chunks))
    if len(conversation_chunks) > 1:
        prompt_sections.append("Conversation history:\n" + "\n".join(conversation_chunks[:-1]))
    prompt_sections.append(latest_user_text)

    return "\n\n".join(section for section in prompt_sections if section).strip()
