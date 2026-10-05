import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, TypedDict

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from agent_runtime.runtime import AgentRuntime
from agent_runtime.worker import Worker
from platform_application.service import Application
from platform_contracts import AgentDescriptor
from platform_infrastructure.artifacts import ArtifactStorage
from platform_infrastructure.sqlite import SQLiteRepository
from platform_access.main import wire_event
from platform_contracts import AgentContext
from platform_sdk.context import use_context


class DirectCoordinator:
    """Exercise real repository fences without network timing in lifecycle tests."""
    def __init__(self, repository, artifacts):
        self.repo, self.artifacts = repository, artifacts
        self.heartbeat = asyncio.Event()

    async def post(self, path, body):
        _, _, rid, operation = path.split("/")
        if operation == "heartbeat":
            result = self.repo.heartbeat(rid, **body)
            self.heartbeat.set()
            return result
        if operation == "artifacts":
            return self.artifacts.upload(rid, **body)
        if operation == "finish":
            cid = self.repo.get_run(rid)["conversation_id"]
            body = {**body, "result": self.artifacts.materialize_result(cid, body.get("result"))}
        name = {"events": "append_event"}.get(operation, operation)
        return getattr(self.repo, name)(rid, **body)

    async def artifact(self, run_id, artifact_id):
        return self.artifacts.get(artifact_id)


def setup(tmp_path, *, execution_class="interactive"):
    repo = SQLiteRepository(tmp_path / "app.sqlite")
    repo.migrate()
    descriptor = AgentDescriptor("fake", "Fake", "Test", "fake.agent", "r1", execution_class=execution_class)
    cid = repo.create_conversation(descriptor, "user", "default")["id"]
    repo.register_worker("generation", execution_class, {"fake": "r1"}, {})
    repo.submit(cid, "user", {"text": "hello"})
    return repo, cid, repo.claim("generation")


def test_cancel_waits_for_blocking_work_and_records_checkpoint(tmp_path):
    async def check():
        repo, cid, run = setup(tmp_path)
        release, entered = asyncio.Event(), asyncio.Event()
        class Runtime:
            cooperative_cancellation = False
            async def execute(self, run, emit):
                entered.set()
                await release.wait()
                return "completed", {"checkpoint_reference": "stopped-checkpoint"}
        client = DirectCoordinator(repo, ArtifactStorage(tmp_path / "api-files", repo))
        worker = Worker(client, Runtime(), generation="generation")
        worker.lease_seconds = 0.03
        task = asyncio.create_task(worker.process(run))
        await entered.wait()
        repo.cancel(run["id"], "user")
        await client.heartbeat.wait()
        assert not task.done()
        assert repo.get_run(run["id"])["status"] == "running"
        release.set()
        await task
        finished = repo.get_run(run["id"])
        assert finished["status"] == "cancelled"
        assert finished["result"]["checkpoint_reference"] == "stopped-checkpoint"
        assert repo.assignment(cid)["checkpoint_reference"] is None
        assert repo.assignment(cid)["blocked_run_id"] == run["id"]
    asyncio.run(check())


def test_output_files_cross_boundary_by_id_and_replay_original_shape(tmp_path):
    async def check():
        repo, cid, run = setup(tmp_path)
        attachment = {"type": "file", "filename": "answer.txt", "content_type": "text/plain",
                      "data": base64.b64encode(b"answer").decode()}
        class Runtime:
            async def execute(self, run, emit):
                return "completed", {"agent_message": {"raw_text": "", "content": {"type": "segments", "parts": [attachment]},
                    "metadata": {"attachments": [attachment]}}}
        class Client(DirectCoordinator):
            async def post(self, path, body):
                if path.endswith("/finish"):
                    assert '"data":' not in json.dumps(body["result"])
                    assert body["result"]["artifacts"]
                return await super().post(path, body)
        storage = ArtifactStorage(tmp_path / "api-files", repo)
        await Worker(Client(repo, storage), Runtime(), generation="generation").process(run)
        result = repo.get_run(run["id"])["result"]
        assert result["agent_message"]["content"]["parts"] == [attachment]
        assert result["agent_message"]["metadata"]["attachments"] == [attachment]
        ref = result["artifacts"][0]
        assert storage.get(ref["id"], "user") == attachment
        assert ref["owner_id"] == "user" and ref["conversation_id"] == cid
        event = wire_event(repo.events(run["id"])[-1], conversation_id=cid)
        assert event["conversation_id"] == cid
        assert "conversation_id" not in event["metadata"]
    asyncio.run(check())


def test_api_repository_restart_and_dropped_notifications_do_not_stop_worker(tmp_path):
    async def check():
        repo, cid, run = setup(tmp_path)
        release, entered = asyncio.Event(), asyncio.Event()
        class Runtime:
            async def execute(self, run, emit):
                entered.set()
                await release.wait()
                await emit({"type": "chunk", "content": "done"})
                return "completed", {"agent_message": {"raw_text": "done", "content": {"type": "text", "text": "done"}, "metadata": {}}}
        task = asyncio.create_task(Worker(DirectCoordinator(repo, ArtifactStorage(tmp_path / "api-files", repo)),
            Runtime(), generation="generation").process(run))
        await entered.wait()
        restarted = SQLiteRepository(repo.path)
        restarted.migrate()
        class DroppedNotifications:
            async def wait(self, timeout):
                await asyncio.sleep(timeout)
            async def publish(self):
                pass
        application = Application(restarted, 0.001, DroppedNotifications())
        detached = application.follow(run["id"], "user")
        await anext(detached)
        await detached.aclose()
        assert not task.done()
        release.set()
        await task
        events = [e async for e in application.follow(run["id"], "user")]
        assert sum(e["type"] == "completed" for e in events) == 1
        assert restarted.conversation(cid, "user", True)["messages"][-1]["raw_text"] == "done"
    asyncio.run(check())


def test_real_graph_interrupt_resume_and_checkpoint_mismatch(tmp_path):
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    def approve(state):
        answer = interrupt({"question": "Approve?", "content": "Draft"})
        return {"messages": [AIMessage(content=f"Approved: {answer}")]}
    async def check():
        manifest = tmp_path / "agents.json"
        manifest.write_text(json.dumps({"agents": [{"id": "fake", "name": "Fake", "description": "Test", "module": "fake.agent", "revision": "r1"}]}))
        runtime = AgentRuntime(str(manifest))
        builder = StateGraph(State)
        builder.add_node("approve", approve)
        builder.add_edge(START, "approve")
        builder.add_edge("approve", END)
        graph = builder.compile(checkpointer=MemorySaver())
        runtime.registry._instances["fake"] = graph
        repo, cid, run = setup(tmp_path)
        client = DirectCoordinator(repo, ArtifactStorage(tmp_path / "api-files", repo))
        worker = Worker(client, runtime, generation="generation")
        await worker.process(run)
        first = repo.get_run(run["id"])
        assert first["status"] == "interrupted", first
        pending = first["result"]["agent_message"]["metadata"]["interrupt_payload"]
        repo.submit(cid, "user", {"text": "yes"}, resume={"interrupt_id": pending["interrupt_id"], "response": "yes"})
        await worker.process(repo.claim("generation"))
        assert repo.conversation(cid, "user", True)["messages"][-1]["raw_text"] == "Approved: yes"
        expected = repo.assignment(cid)["checkpoint_reference"]
        await graph.aupdate_state({"configurable": {"thread_id": cid}}, {"messages": [AIMessage(content="uncommitted")]})
        next_run = repo.submit(cid, "user", {"text": "next"})
        await worker.process(repo.claim("generation"))
        assert repo.get_run(next_run["id"])["status"] == "recovery_required"
        assert repo.assignment(cid)["checkpoint_reference"] == expected
        await runtime.close()
    asyncio.run(check())


def test_privacy_child_requires_affinity_on_its_root_assignment(tmp_path):
    manifest = tmp_path / "agents.json"
    manifest.write_text(json.dumps({"agents": [
        {"id": "root", "name": "Root", "description": "Root", "module": "fake.root"},
        {"id": "child", "name": "Child", "description": "Child", "module": "fake.child", "privacy_affinity": True}]}))
    async def check():
        runtime = AgentRuntime(str(manifest))
        runtime.registry._instances["child"] = object()
        with use_context(AgentContext("user", "default", "conversation", "run", trace={"agent_id": "root"})):
            with pytest.raises(ValueError, match="must declare privacy_affinity"):
                await runtime.resolve("child", state_scope="stateful")
        await runtime.close()
    asyncio.run(check())
