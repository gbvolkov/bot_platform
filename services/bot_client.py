from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class BotServiceClient:
    def __init__(
        self,
        *,
        base_url: str,
        request_timeout: float = 180.0,
        connect_timeout: float = 10.0,
    ) -> None:
        timeout = httpx.Timeout(request_timeout, connect=connect_timeout)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            follow_redirects=True,
        )
        self._agents: Dict[str, Dict[str, Any]] = {}

    async def startup(self) -> None:
        await self.refresh_agents()

    async def shutdown(self) -> None:
        await self._client.aclose()

    async def refresh_agents(self) -> None:
        response = await self._client.get("/agents/")
        response.raise_for_status()
        agents = response.json()
        self._agents = {agent["id"]: agent for agent in agents}

    async def ensure_agent(self, agent_id: str) -> None:
        if agent_id in self._agents:
            return
        await self.refresh_agents()
        if agent_id not in self._agents:
            raise KeyError(f"Unknown agent '{agent_id}'")

    async def list_agents(self) -> list[Dict[str, Any]]:
        if not self._agents:
            await self.refresh_agents()
        return list(self._agents.values())

    async def create_conversation(
        self,
        agent_id: str,
        user_id: str,
        user_role: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"agent_id": agent_id}
        if user_role:
            payload["user_role"] = user_role
        if title:
            payload["title"] = title
        headers = {"X-User-Id": user_id}
        if user_role:
            headers["X-User-Role"] = user_role

        response = await self._client.post("/conversations/", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    async def send_message(
        self,
        conversation_id: str,
        user_id: str,
        text: str,
        user_role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "payload": {
                "type": "text",
                "text": text,
                "metadata": metadata or {},
            }
        }
        headers = {"X-User-Id": user_id}
        if user_role:
            headers["X-User-Role"] = user_role

        response = await self._client.post(
            f"/conversations/{conversation_id}/messages",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
