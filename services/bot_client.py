from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import httpx
import logging


logger = logging.getLogger(__name__)


def _redact_attachment(attachment: Dict[str, Any]) -> Dict[str, Any]:
    redacted = dict(attachment)
    data = redacted.get("data")
    if isinstance(data, str):
        redacted["data"] = f"<base64 {len(data)} chars>"
    return redacted


def _redact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = dict(payload)
    message_payload = redacted.get("payload")
    if isinstance(message_payload, dict):
        msg_copy: Dict[str, Any] = dict(message_payload)
        attachments = msg_copy.get("attachments")
        if isinstance(attachments, list):
            msg_copy["attachments"] = [
                _redact_attachment(att) if isinstance(att, dict) else att for att in attachments
            ]
        redacted["payload"] = msg_copy
    return redacted


class BotServiceClient:
    def __init__(
        self,
        *,
        base_url: str,
        request_timeout: float | None = 180.0,
        connect_timeout: float = 10.0,
    ) -> None:
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=None,
            write=None,
            pool=None,
        )
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
        await self.refresh_agents()
        return list(self._agents.values())

    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        await self.ensure_agent(agent_id)
        return self._agents[agent_id]

    async def create_conversation(
        self,
        agent_id: str,
        user_id: str,
        user_role: Optional[str] = None,
        title: Optional[str] = None,
    ) -> tuple[Dict[str, Any], bool]:
        payload: Dict[str, Any] = {"agent_id": agent_id}
        if user_role:
            payload["user_role"] = user_role
        if title:
            payload["title"] = title
        headers = {"X-User-Id": user_id}
        if user_role:
            headers["X-User-Role"] = user_role
        logger.info("POST /conversations payload=%s headers=%s", payload, {"X-User-Id": headers["X-User-Id"], "X-User-Role": headers.get("X-User-Role")})
        response = await self._client.post("/conversations/", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data, response.status_code == httpx.codes.CREATED

    async def send_message(
        self,
        conversation_id: str,
        user_id: str,
        text: str,
        raw_user_text: Optional[str] = None,
        user_role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        attachments: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload_metadata: Dict[str, Any] = metadata.copy() if isinstance(metadata, dict) else {}
        if raw_user_text and "raw_user_text" not in payload_metadata:
            payload_metadata["raw_user_text"] = raw_user_text
        payload = {
            "payload": {
                "type": "text",
                "text": text,
                "metadata": payload_metadata,
            }
        }
        if attachments:
            payload["payload"]["attachments"] = list(attachments)
        headers = {"X-User-Id": user_id}
        if user_role:
            headers["X-User-Role"] = user_role
        logger.info(
            "POST /conversations/%s/messages payload=%s headers=%s",
            conversation_id,
            _redact_payload(payload),
            {"X-User-Id": headers["X-User-Id"], "X-User-Role": headers.get("X-User-Role")},
        )

        response = await self._client.post(
            f"/conversations/{conversation_id}/messages",
            json=payload,
            headers=headers,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "POST /conversations/%s/messages failed status=%s body=%s",
                conversation_id,
                exc.response.status_code if exc.response else None,
                exc.response.text if exc.response else None,
            )
            raise
        return response.json()

    async def get_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        headers = {"X-User-Id": user_id}
        response = await self._client.get(f"/conversations/{conversation_id}", headers=headers)
        response.raise_for_status()
        return response.json()
