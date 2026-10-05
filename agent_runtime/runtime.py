"""Agent lifecycle and local/hosted LangGraph execution composition."""
import asyncio
import importlib
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

from platform_contracts import AgentContext, RunRequest, RunResult, ArtifactError
from platform_contracts.catalog import AgentCatalog, DEFAULT_CATALOG_PATH
from platform_sdk.schemas import MessagePayload
from platform_sdk.context import use_context, current_context
from .invocation import invoke_agent, invoke_agent_stream, serialise_message
from .registry import AgentRegistry
from .config import settings

_loading_agents = ContextVar("loading_agents", default=())


class AgentRuntime:
    # Some graph nodes execute blocking tools on threads. Cancelling their
    # asyncio wrapper cannot stop the tools. A root turn is the safe boundary
    # until a plugin explicitly supports an earlier cooperative stop.
    cooperative_cancellation = False

    def __init__(self, catalog_path=DEFAULT_CATALOG_PATH, *, capabilities=None):
        self.catalog = AgentCatalog.load(catalog_path)
        self.capabilities = capabilities or {}
        self.registry = AgentRegistry(catalog_path, resolver=self, capabilities=self.capabilities)
        self.loop = None
        self._loading = {}

    def describe(self, agent_id):
        return self.catalog.agents[agent_id]

    async def resolve(self, agent_id, *, state_scope):
        if state_scope not in {"stateless", "stateful"}:
            raise ValueError("Child state scope must be explicit")
        # Synchronous legacy factories run in an executor and may enter their
        # own event loop. Checkpointers must stay on the owning runtime loop.
        if self.loop is not None and asyncio.get_running_loop() is not self.loop:
            future = asyncio.run_coroutine_threadsafe(self.compiled(agent_id, allow_inactive=True), self.loop)
            return await asyncio.wrap_future(future)
        return await self.compiled(agent_id, allow_inactive=True)

    async def compiled(self, agent_id, *, allow_inactive=False):
        self.loop = asyncio.get_running_loop()
        descriptor = self.describe(agent_id) if allow_inactive else self.catalog.get(agent_id)
        missing = set(descriptor.required_services) - set(self.capabilities)
        if missing:
            raise ValueError(f"Missing required services for {agent_id}: {sorted(missing)}")
        chain = _loading_agents.get()
        context = current_context()
        parent = chain[0] if chain else (context.trace.get("agent_id") if context else None)
        if parent and parent != agent_id and descriptor.privacy_affinity and not self.describe(parent).privacy_affinity:
            raise ValueError(f"Agent '{parent}' must declare privacy_affinity to delegate to '{agent_id}'")
        if agent_id in chain:
            raise ValueError(f"Circular agent dependency: {' -> '.join((*chain, agent_id))}")
        token = _loading_agents.set((*chain, agent_id))
        try:
            while not await self.registry.ensure_agent_ready(agent_id):
                await asyncio.sleep(0.01)
            return self.registry.get_agent(agent_id)
        finally:
            _loading_agents.reset(token)

    async def initialize(self, execution_class, progress=None):
        ready, errors = {}, {}
        async def initialize_one(descriptor):
            try:
                await self.compiled(descriptor.id)
                ready[descriptor.id] = descriptor.revision
            except Exception as exc:
                errors[descriptor.id] = f"{type(exc).__name__}: {exc}"
            if progress is not None:
                await progress(dict(ready), dict(errors))
        await asyncio.gather(*(initialize_one(d) for d in self.catalog.agents.values()
                               if d.active and d.execution_class == execution_class))
        return ready, errors

    async def prepare(self, agent_id, raw):
        payload = MessagePayload.model_validate(raw)
        payload.metadata = dict(payload.metadata)
        # Caller-supplied host paths are never used for new-path attachments.
        payload.metadata.pop("raw_attachments", None)
        payload.metadata.pop("attachment_text_segments", None)
        if payload.attachments:
            from platform_capabilities.ingestion import process_attachments, attachment_to_text_segment
            allow_raw = self.registry.allows_raw_attachments(agent_id)
            try:
                processed = await asyncio.to_thread(process_attachments, payload.attachments,
                    self.registry.supported_content_types(agent_id), persist_raw=allow_raw,
                    persist_dir=Path(settings.attachment_store_dir).resolve() if allow_raw else None)
            except Exception as exc:
                raise ArtifactError(f"Attachment processing failed: {exc}") from exc
            failed = [p.attachment.filename for p in processed if p.error or (not p.supported and not p.text)]
            if failed:
                raise ArtifactError(f"Failed to process unsupported attachments: {', '.join(failed)}")
            payload.metadata["attachments"] = [p.as_metadata() for p in processed]
            payload.metadata["attachment_text_segments"] = [segment for p in processed if (segment := attachment_to_text_segment(p))]
            if allow_raw:
                payload.metadata["raw_attachments"] = [{"filename": p.attachment.filename,
                    "content_type": p.attachment.content_type, "path": p.stored_path} for p in processed if p.stored_path]
        return payload

    @staticmethod
    def result(value):
        message = serialise_message(value["ai"])
        metadata = {"agent_status": value["agent_status"]}
        if message["attachments"]:
            metadata["attachments"] = message["attachments"]
        if value.get("interrupt_payload"):
            metadata.update(interrupt_payload=value["interrupt_payload"], question=message["raw_text"], content=message["raw_text"])
        return {"agent_message": {"content": message["content"], "raw_text": message["raw_text"], "metadata": metadata}}

    async def execute(self, run, emit):
        context = AgentContext(run["user_id"], run["user_role"], run["conversation_id"], run["id"],
            parent_run_id=run.get("parent_run_id"), trace={"attempt_id": run.get("attempt_id") or "", "agent_id": run["agent_id"]},
            capabilities=self.capabilities)
        with use_context(context):
            return await self._execute(run, emit)

    async def _execute(self, run, emit):
        agent = await self.compiled(run["agent_id"])
        config = {"configurable": {"thread_id": run["conversation_id"], "user_id": run["user_id"], "user_role": run["user_role"]}}
        async def checkpoint():
            if not hasattr(agent, "aget_state"):
                return None
            state = await agent.aget_state(config)
            return (state.config or {}).get("configurable", {}).get("checkpoint_id")
        before = await checkpoint()
        if before != run.get("expected_checkpoint"):
            return "recovery_required", {"checkpoint_reference": before,
                "error": "Checkpoint does not match the last committed application result"}
        payload = await self.prepare(run["agent_id"], run["input"])
        events, future = await invoke_agent_stream(agent, payload, run["conversation_id"], run["agent_id"],
            run["user_id"], run["user_role"], pending_interrupt=run["input"].get("pending_interrupt"), registry=self.registry)
        delivery_error = None
        try:
            async for event in events:
                if delivery_error is None:
                    try:
                        await emit(event)
                    except Exception as exc:
                        # Drain execution before releasing the root slot, even
                        # if persistence becomes unavailable during a tool call.
                        delivery_error = exc
            result = await future
            if delivery_error is not None:
                raise delivery_error
            output = self.result(result)
            human = serialise_message(result["human"])
            metadata = dict(payload.metadata)
            # Local raw paths belong exclusively to the worker storage adapter.
            for raw, reference in zip(metadata.get("raw_attachments", []), run["input"].get("artifact_refs", [])):
                raw["path"] = f"/api/artifacts/{reference}"
            output["accepted_message"] = {"content": human["content"], "raw_text": human["raw_text"], "metadata": metadata}
            if hasattr(agent, "aget_state"):
                output["checkpoint_reference"] = await checkpoint()
            exporter = self.describe(run["agent_id"]).configuration.get("artifact_exporter")
            if exporter:
                module, name = exporter.split(":", 1)
                collect = getattr(importlib.import_module(module), name)
                state = await agent.aget_state(config)
                try:
                    output["artifact_payloads"] = await asyncio.to_thread(collect, state.values)
                except Exception as exc:
                    raise ArtifactError(f"Artifact export failed: {exc}") from exc
            return result["agent_status"], output
        finally:
            await events.aclose()

    async def close(self):
        await self.registry.aclose()

    def handle(self, agent_id):
        return LangGraphHandle(self, agent_id)


class LangGraphHandle:
    def __init__(self, runtime, agent_id):
        self.runtime, self.agent_id = runtime, agent_id

    def _context(self, context):
        return replace(context, trace={**context.trace, "agent_id": self.agent_id},
                       capabilities={**self.runtime.capabilities, **context.capabilities})

    @staticmethod
    async def _checkpoint(agent, context):
        if not hasattr(agent, "aget_state"):
            return None
        state = await agent.aget_state({"configurable": {"thread_id": context.conversation_id,
            "user_id": context.user_id, "user_role": context.user_role}})
        return (state.config or {}).get("configurable", {}).get("checkpoint_id")

    async def invoke(self, request: RunRequest, context: AgentContext) -> RunResult:
        descriptor = self.runtime.describe(self.agent_id)
        if request.agent_revision != descriptor.revision:
            raise ValueError("Agent revision mismatch")
        with use_context(self._context(context)):
            agent = await self.runtime.compiled(self.agent_id)
            payload = await self.runtime.prepare(self.agent_id, request.input)
            value = await invoke_agent(agent, payload, context.conversation_id, self.agent_id,
                context.user_id, context.user_role, pending_interrupt=request.input.get("pending_interrupt"), registry=self.runtime.registry)
            checkpoint = await self._checkpoint(agent, context)
        return RunResult(self.runtime.result(value), interrupt=value.get("interrupt_payload"), checkpoint=checkpoint)

    async def stream(self, request, context):
        from platform_contracts import RunEvent
        if request.agent_revision != self.runtime.describe(self.agent_id).revision:
            raise ValueError("Agent revision mismatch")
        # The producer inherits this context. Restore the caller's context before
        # yielding so an attached client cannot inherit the agent's identity.
        with use_context(self._context(context)):
            agent = await self.runtime.compiled(self.agent_id)
            payload = await self.runtime.prepare(self.agent_id, request.input)
            events, future = await invoke_agent_stream(agent, payload, context.conversation_id, self.agent_id,
                context.user_id, context.user_role, pending_interrupt=request.input.get("pending_interrupt"), registry=self.runtime.registry)
        sequence = 0
        try:
            async for event in events:
                sequence += 1
                yield RunEvent(context.run_id, None, sequence, event["type"], event)
            result = await future
            output = self.runtime.result(result)
            output["checkpoint_reference"] = await self._checkpoint(agent, context)
            yield RunEvent(context.run_id, None, sequence+1, result["agent_status"], output)
        finally:
            await events.aclose()

    async def resume(self, interrupt_id, response, context):
        return await self.invoke(RunRequest("resume", {"text": response, "pending_interrupt": {"interrupt_id": interrupt_id}},
                                 self.runtime.describe(self.agent_id).revision, 0), context)

    async def reset(self, context):
        return await self.invoke(RunRequest("reset", {"type": "reset"}, self.runtime.describe(self.agent_id).revision, 0), context)

    async def close(self):
        # Runtime owns the shared plugin and checkpoint lifecycle.
        return None
