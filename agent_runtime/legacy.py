"""Intentional bridge to the unchanged live legacy host, never a replacement."""
import httpx


class LegacyRuntime:
    cooperative_cancellation = False
    cancel_at_completion = False  # The unchanged host owns its result projection.
    def __init__(self, url):
        self.client = httpx.AsyncClient(base_url=url, timeout=None)

    async def initialize(self, execution_class, progress=None):
        return {}, {}

    async def execute(self, run, emit):
        payload = {key: value for key, value in run["input"].items() if key in {"type", "text", "metadata", "attachments"}}
        import json
        headers = {"X-User-Id": run["user_id"], "X-User-Role": run["user_role"]}
        terminal = None
        async with self.client.stream("POST", f"/conversations/{run['conversation_id']}/messages?stream=true",
                                      json={"payload": payload}, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                event = json.loads(line[5:])
                if event["type"] in {"chunk", "custom"}:
                    await emit(event)
                elif event["type"] in {"completed", "interrupt"}:
                    terminal = event
                elif event["type"] == "failed":
                    raise RuntimeError(event.get("error", "Legacy execution failed"))
        if terminal is None:
            raise RuntimeError("Legacy stream ended without a terminal result; reconcile before retrying")
        detail = await self.client.get(f"/conversations/{run['conversation_id']}", headers=headers)
        detail.raise_for_status()
        conversation = detail.json()
        messages = conversation.pop("messages")
        result = {"conversation": conversation, "user_message": messages[-2], "agent_message": messages[-1]}
        # Legacy owns both message projections. A lost HTTP response requires
        # explicit recovery; the coordinator never blindly retries this call.
        return result["agent_message"].get("metadata", {}).get("agent_status", "completed"), result

    async def close(self):
        await self.client.aclose()
