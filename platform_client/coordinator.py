import asyncio

import httpx


class CoordinatorClient:
    def __init__(self, url, token):
        if not token:
            raise ValueError("PLATFORM_SERVICE_TOKEN is required")
        self.client = httpx.AsyncClient(base_url=url, headers={"Authorization": "Bearer " + token}, timeout=10)

    async def post(self, path, body):
        # These coordinator operations are fenced/idempotent. Retrying a lost
        # response never repeats agent execution.
        for attempt in range(3):
            try:
                response = await self.client.post(path, json=body)
                response.raise_for_status()
                return response.json()
            except httpx.TransportError:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.25 * (attempt+1))

    async def artifact(self, run_id, artifact_id):
        response = await self.client.get(f"/runs/{run_id}/artifacts/{artifact_id}")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()
