import asyncio
import base64
import json

import httpx

from agent_runtime.worker import Worker
from platform_access.main import create_app
from platform_access.coordinator import create_app as create_coordinator
from platform_client.coordinator import CoordinatorClient
from platform_client.queue_adapter import DurableQueueClient
from platform_infrastructure.settings import PlatformSettings
from platform_infrastructure.sqlite import SQLiteRepository
from services.task_queue.models import EnqueuePayload


class FakeRuntime:
    def __init__(self):
        self.inputs = []
    async def execute(self, run, emit):
        self.inputs.append(run)
        await emit({"type": "chunk", "content": "Hello "})
        await emit({"type": "chunk", "content": "world"})
        return "completed", {"agent_message": {"content": {"type": "text", "text": "Hello world"}, "raw_text": "Hello world", "metadata": {}}}


def test_public_api_worker_and_proxy_share_durable_runs(tmp_path):
    async def run():
        manifest = tmp_path / "agents.json"
        manifest.write_text(json.dumps({"agents": [{"id": "fake", "name": "Fake", "description": "Test", "module": "not_installed.agent", "revision": "r1"}]}))
        settings = PlatformSettings(_env_file=None, database_path=str(tmp_path / "app.sqlite"),
            catalog_path=str(manifest), artifact_path=str(tmp_path / "api-only-artifacts"), service_token="test-credential",
            worker_agent_ids="*", poll_seconds=0.001)
        repo = SQLiteRepository(settings.database_path)
        repo.migrate()
        public = create_app(settings, repo)
        coordinator = create_coordinator(settings, repo)
        api = httpx.AsyncClient(transport=httpx.ASGITransport(app=public), base_url="http://api")
        private = CoordinatorClient("http://coordinator", settings.service_token)
        await private.client.aclose()
        private.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=coordinator), base_url="http://coordinator", headers={"Authorization": "Bearer test-credential"})
        runtime = FakeRuntime()
        worker = Worker(private, runtime, generation="g1")
        worker.ready = {"fake": "r1"}
        await worker.register()
        assert (await api.get("/api/agents/")).json()[0]["id"] == "fake"
        assert (await api.post("/runs/claim", json={"generation": "g1"})).status_code == 404
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=coordinator), base_url="http://private") as unauthorized:
            assert (await unauthorized.post("/runs/claim", json={"generation": "g1"})).status_code == 401
        conversation = (await api.post("/api/conversations/", json={"agent_id": "fake"}, headers={"X-User-Id": "user"})).json()
        cid = conversation["id"]
        queue = DurableQueueClient("http://api/api", transport=httpx.ASGITransport(app=public))
        payload = EnqueuePayload(job_id="j1", model="fake", conversation_id=cid, user_id="user", text="hi", stream=True,
            attachments=[{"filename": "test.txt", "data": base64.b64encode(b"attachment").decode()}])
        await queue.enqueue(payload)
        await queue.enqueue(payload)
        claim = await private.post("/runs/claim", {"generation": "g1"})
        assert "attachments" not in claim["input"]
        assert claim["input"]["artifact_refs"]
        await worker.process(claim)
        # Subscribe after completion, as a restarted/detached API client would.
        received = [event async for event in queue.iter_events("j1")]
        assert [e.type for e in received] == ["chunk", "chunk", "completed"]
        assert "content" not in received[-1].metadata
        assert runtime.inputs[0]["input"]["attachments"][0]["data"] == base64.b64encode(b"attachment").decode()
        assert len(repo.conversation(cid, "user", True)["messages"]) == 2
        assert (await api.get(f"/api/runs/{claim['id']}", headers={"X-User-Id": "other"})).status_code == 404
        result = (await api.get(f"/api/runs/{claim['id']}", headers={"X-User-Id": "user"})).json()
        assert result["result"]["agent_message"]["raw_text"] == "Hello world"
        # Existing synchronous and SSE adapters retain their public shapes.
        async def send_native(stream):
            pending = asyncio.create_task(api.post(f"/api/conversations/{cid}/messages", params={"stream": str(stream).lower()},
                json={"payload": {"text": "next"}}, headers={"X-User-Id": "user"}))
            claimed = None
            for _ in range(100):
                claimed = await private.post("/runs/claim", {"generation": "g1"})
                if claimed:
                    break
                await asyncio.sleep(0.01)
            assert claimed is not None
            await worker.process(claimed)
            return await pending
        response = await send_native(False)
        assert response.status_code == 201
        assert set(response.json()) == {"conversation", "user_message", "agent_message"}
        assert "T" in response.json()["conversation"]["created_at"]
        response = await send_native(True)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        assert [e["type"] for e in events] == ["chunk", "chunk", "completed"]
        assert all(e["conversation_id"] == cid for e in events)
        assert "conversation_id" not in events[-1]["metadata"]
        assert response.text.endswith("data: [DONE]\n\n")
        await api.aclose()
        await private.close()
        await queue.shutdown()
    asyncio.run(run())
