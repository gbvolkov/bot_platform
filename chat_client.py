from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx
import typer

app = typer.Typer(help="Command-line client for the bot platform API.")

API_BASE_URL = os.environ.get("BOT_SERVICE_BASE_URL", "http://localhost:8000/api")


def _api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}" if path.startswith("/") else f"{API_BASE_URL.rstrip('/')}/{path}"  # noqa: E501


async def _list_agents() -> None:
    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(_api_url("/agents/"))
        response.raise_for_status()
        agents = response.json()
        if not agents:
            typer.echo("No agents registered.")
            return
        typer.echo("Available agents:\n")
        for agent in agents:
            typer.echo(f"- {agent['id']}: {agent['name']} — {agent['description']}")


@app.command()
def list_agents() -> None:
    """Show all available agents."""

    asyncio.run(_list_agents())


async def _create_conversation(agent_id: str, title: Optional[str], user_role: Optional[str]) -> str:
    payload = {"agent_id": agent_id}
    if title:
        payload["title"] = title
    if user_role:
        payload["user_role"] = user_role

    headers = {"X-User-Id": os.environ.get("BOT_SERVICE_USER_ID", "cli-user")}
    if user_role:
        headers["X-User-Role"] = user_role

    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.post(_api_url("/conversations/"), json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        typer.echo(f"Started conversation {data['id']} with agent '{data['agent_id']}'.")
        return data["id"]


async def _send_message(conversation_id: str, text: str, *, reset: bool = False) -> None:
    payload = {"payload": {"type": "reset" if reset else "text"}}
    if not reset:
        payload["payload"]["text"] = text
    else:
        payload["payload"]["text"] = text or "RESET"

    headers = {"X-User-Id": os.environ.get("BOT_SERVICE_USER_ID", "cli-user")}
    user_role = os.environ.get("BOT_SERVICE_USER_ROLE")
    if user_role:
        headers["X-User-Role"] = user_role

    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.post(
            _api_url(f"/conversations/{conversation_id}/messages"),
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
        agent_message = data["agent_message"]["raw_text"].strip()
        if agent_message:
            typer.echo(f"\nAssistant: {agent_message}\n")
        else:
            typer.echo("\nAssistant cleared memory.\n")


@app.command()
def chat(
    agent_id: str = typer.Argument(..., help="Agent identifier to talk to."),
    conversation_id: Optional[str] = typer.Option(None, help="Reuse an existing conversation."),
    title: Optional[str] = typer.Option(None, help="Optional title for a new conversation."),
    user_role: Optional[str] = typer.Option(None, help="Role to pass to the agent."),
) -> None:
    """Interactive chat session with a given agent."""

    async def _chat() -> None:
        conv_id = conversation_id
        if conv_id is None:
            conv_id = await _create_conversation(agent_id, title, user_role)
        typer.echo("Type /reset to clear memory or /exit to leave.\n")
        while True:
            user_input = typer.prompt("You")
            if user_input.strip().lower() in {"/exit", "exit"}:
                typer.echo("Goodbye!")
                break
            if user_input.strip().lower() == "/reset":
                await _send_message(conv_id, "RESET", reset=True)
                continue
            await _send_message(conv_id, user_input, reset=False)

    asyncio.run(_chat())


if __name__ == "__main__":
    app()
