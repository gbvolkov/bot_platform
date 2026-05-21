from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langgraph.graph import StateGraph

from .graph_spec import AgentGraphSpec, AgentNodeSpec, CallableNodeSpec, NodeGuardrailPolicy
from .middleware import guarded_node
from .runtime import PlatformGuardrailRuntime


class PlatformGraphCompiler:
    """Compile AgentGraphSpec into a guarded LangGraph runnable."""

    def compile(
        self,
        spec: AgentGraphSpec,
        *,
        guardrail_runtime: PlatformGuardrailRuntime | None = None,
        checkpointer: Any = None,
        tools: list[Any] | None = None,
        tool_profiles: dict[str, Any] | None = None,
    ) -> Any:
        runtime = guardrail_runtime or PlatformGuardrailRuntime.disabled(agent_id="unknown")
        builder = StateGraph(
            spec.state_schema,
            spec.context_schema,
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
            **spec.graph_kwargs,
        )
        platform_tools = list(tools or [])
        profiles = dict(tool_profiles or {})

        for node in spec.nodes:
            if isinstance(node, CallableNodeSpec):
                builder.add_node(
                    node.name,
                    self._compile_callable_node(node, runtime),
                    **node.add_node_kwargs,
                )
                continue
            if isinstance(node, AgentNodeSpec):
                builder.add_node(
                    node.name,
                    self._compile_agent_node(
                        node,
                        runtime,
                        tools=platform_tools,
                        tool_profiles=profiles,
                    ),
                    **node.add_node_kwargs,
                )
                continue
            raise TypeError(f"Unsupported graph node spec: {node!r}")

        for conditional in spec.conditional_edges:
            builder.add_conditional_edges(
                conditional.source,
                conditional.path,
                conditional.path_map,
            )
        for edge in spec.edges:
            builder.add_edge(edge.start_key, edge.end_key)

        compile_options = dict(spec.compile_options)
        compiled = builder.compile(
            checkpointer=checkpointer,
            debug=bool(compile_options.pop("debug", False)),
            **compile_options,
        )
        if spec.callbacks:
            compiled = compiled.with_config({"callbacks": list(spec.callbacks)})
        return compiled

    def _compile_callable_node(
        self,
        node: CallableNodeSpec,
        runtime: PlatformGuardrailRuntime,
    ) -> Any:
        policy = node.guardrails
        if not policy.enabled:
            return node.action
        scan_state_keys = runtime.state_keys_for_policy(policy.scan_state_keys)
        security = (
            runtime.security_middleware(
                agent_name=f"{runtime.agent_id}.{node.name}",
                scan_state_keys=scan_state_keys,
                composite_input_scanners=policy.composite_input_scanners,
                composite_recent_message_limit=policy.composite_recent_message_limit,
                composite_message_roles=policy.composite_message_roles,
            )
            if policy.scan_input or policy.scan_output
            else None
        )
        privacy = (
            runtime.privacy_middleware(
                agent_name=f"{runtime.agent_id}.{node.name}",
            )
            if policy.privacy
            else None
        )
        if security is None and privacy is None:
            return node.action
        return guarded_node(
            node.action,
            security_middleware=security if policy.scan_input or policy.scan_output else None,
            privacy_middleware=privacy,
            scan_output=policy.scan_output,
            scan_state_keys=scan_state_keys,
            privacy_state_keys=policy.scan_state_keys,
            composite_input_scanners=policy.composite_input_scanners,
            composite_recent_message_limit=policy.composite_recent_message_limit,
            composite_message_roles=policy.composite_message_roles,
        )

    def _compile_agent_node(
        self,
        node: AgentNodeSpec,
        runtime: PlatformGuardrailRuntime,
        *,
        tools: list[Any],
        tool_profiles: dict[str, Any],
    ) -> Any:
        policy = node.guardrails
        resolved_tools = tools if node.tools_source == "platform" else list(node.tools)
        scan_state_keys = runtime.state_keys_for_policy(policy.scan_state_keys)
        middleware: list[Any] = []
        if node.prompt is not None:
            middleware.append(node.prompt)

        if policy.enabled:
            security = runtime.security_middleware(
                agent_name=f"{runtime.agent_id}.{node.name}",
                scan_system_prompt=True,
                scan_state_keys=scan_state_keys,
                composite_input_scanners=policy.composite_input_scanners,
                composite_recent_message_limit=policy.composite_recent_message_limit,
                composite_message_roles=policy.composite_message_roles,
            )
            privacy = runtime.privacy_middleware(
                agent_name=f"{runtime.agent_id}.{node.name}",
                guard_tool_calls=False,
            )
            tool_execution = runtime.tool_execution_middleware(
                tools=list(resolved_tools),
                tool_profiles=tool_profiles,
                agent_name=f"{runtime.agent_id}.{node.name}",
            )
            middleware.extend(_present([security, privacy, tool_execution]))

        middleware.extend(node.middleware)
        return create_agent(
            model=node.model,
            tools=list(resolved_tools),
            middleware=middleware,
            response_format=node.response_format,
            state_schema=node.state_schema,
            context_schema=node.context_schema,
            **node.create_agent_kwargs,
        )


def _present(values: list[Any | None]) -> list[Any]:
    return [value for value in values if value is not None]


__all__ = ["PlatformGraphCompiler"]
