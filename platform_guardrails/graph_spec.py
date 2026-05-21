from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal


GuardrailSetting = Any
ToolsSource = Literal["platform", "node"]


@dataclass(frozen=True)
class NodeGuardrailPolicy:
    enabled: bool = True
    scan_input: bool = True
    scan_output: bool = True
    privacy: bool = True
    scan_state_keys: tuple[str, ...] = ("system_prompt",)
    composite_input_scanners: tuple[str, ...] | None = ("PromptInjection",)
    composite_recent_message_limit: int = 20
    composite_message_roles: tuple[str, ...] | None = None

    @classmethod
    def disabled(cls) -> "NodeGuardrailPolicy":
        return cls(enabled=False, scan_input=False, scan_output=False, privacy=False)


@dataclass(frozen=True)
class CallableNodeSpec:
    name: str
    action: Callable[..., Any]
    guardrails: NodeGuardrailPolicy
    add_node_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentNodeSpec:
    name: str
    model: Any
    prompt: Any | None = None
    tools_source: ToolsSource = "platform"
    tools: tuple[Any, ...] = ()
    middleware: tuple[Any, ...] = ()
    state_schema: type | None = None
    context_schema: type | None = None
    response_format: Any | None = None
    create_agent_kwargs: dict[str, Any] = field(default_factory=dict)
    guardrails: NodeGuardrailPolicy = field(default_factory=NodeGuardrailPolicy)
    add_node_kwargs: dict[str, Any] = field(default_factory=dict)


GraphNodeSpec = CallableNodeSpec | AgentNodeSpec


@dataclass(frozen=True)
class EdgeSpec:
    start_key: str | list[str]
    end_key: str


@dataclass(frozen=True)
class ConditionalEdgeSpec:
    source: str
    path: Any
    path_map: dict[Any, str] | list[str] | None = None


@dataclass(frozen=True)
class AgentGraphSpec:
    state_schema: type
    context_schema: type | None = None
    input_schema: type | None = None
    output_schema: type | None = None
    graph_kwargs: dict[str, Any] = field(default_factory=dict)
    nodes: tuple[GraphNodeSpec, ...] = ()
    edges: tuple[EdgeSpec, ...] = ()
    conditional_edges: tuple[ConditionalEdgeSpec, ...] = ()
    compile_options: dict[str, Any] = field(default_factory=dict)
    callbacks: tuple[Any, ...] = ()
    default_node_policy: NodeGuardrailPolicy = field(default_factory=NodeGuardrailPolicy)


def coerce_node_guardrail_policy(
    value: GuardrailSetting | None,
    *,
    default: NodeGuardrailPolicy,
) -> NodeGuardrailPolicy:
    if value is None:
        return default
    if isinstance(value, NodeGuardrailPolicy):
        return value
    if isinstance(value, bool):
        return default if value else NodeGuardrailPolicy.disabled()
    raise TypeError("guardrails must be a boolean or NodeGuardrailPolicy.")


class PlatformStateGraph:
    """Collect a graph definition for platform compilation."""

    def __init__(
        self,
        state_schema: type,
        context_schema: type | None = None,
        *,
        input_schema: type | None = None,
        output_schema: type | None = None,
        default_node_policy: NodeGuardrailPolicy | None = None,
        **graph_kwargs: Any,
    ) -> None:
        self._state_schema = state_schema
        self._context_schema = context_schema
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._graph_kwargs = dict(graph_kwargs)
        self._default_node_policy = default_node_policy or NodeGuardrailPolicy()
        self._nodes: list[GraphNodeSpec] = []
        self._edges: list[EdgeSpec] = []
        self._conditional_edges: list[ConditionalEdgeSpec] = []

    def add_node(
        self,
        name: str,
        node: Callable[..., Any],
        *,
        guardrails: GuardrailSetting | None = None,
        **add_node_kwargs: Any,
    ) -> "PlatformStateGraph":
        self._nodes.append(
            CallableNodeSpec(
                name=name,
                action=node,
                guardrails=coerce_node_guardrail_policy(
                    guardrails,
                    default=self._default_node_policy,
                ),
                add_node_kwargs=dict(add_node_kwargs),
            )
        )
        return self

    def add_agent_node(
        self,
        name: str,
        *,
        model: Any,
        prompt: Any | None = None,
        tools_source: ToolsSource = "platform",
        tools: Iterable[Any] = (),
        middleware: Iterable[Any] = (),
        state_schema: type | None = None,
        context_schema: type | None = None,
        response_format: Any | None = None,
        guardrails: GuardrailSetting | None = None,
        create_agent_kwargs: dict[str, Any] | None = None,
        **add_node_kwargs: Any,
    ) -> "PlatformStateGraph":
        self._nodes.append(
            AgentNodeSpec(
                name=name,
                model=model,
                prompt=prompt,
                tools_source=tools_source,
                tools=tuple(tools),
                middleware=tuple(middleware),
                state_schema=state_schema,
                context_schema=context_schema,
                response_format=response_format,
                create_agent_kwargs=dict(create_agent_kwargs or {}),
                guardrails=coerce_node_guardrail_policy(
                    guardrails,
                    default=self._default_node_policy,
                ),
                add_node_kwargs=dict(add_node_kwargs),
            )
        )
        return self

    def add_edge(self, start_key: str | list[str], end_key: str) -> "PlatformStateGraph":
        self._edges.append(EdgeSpec(start_key=start_key, end_key=end_key))
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Any,
        path_map: dict[Any, str] | list[str] | None = None,
    ) -> "PlatformStateGraph":
        self._conditional_edges.append(
            ConditionalEdgeSpec(source=source, path=path, path_map=path_map)
        )
        return self

    def to_spec(
        self,
        *,
        callbacks: Iterable[Any] = (),
        compile_options: dict[str, Any] | None = None,
    ) -> AgentGraphSpec:
        return AgentGraphSpec(
            state_schema=self._state_schema,
            context_schema=self._context_schema,
            input_schema=self._input_schema,
            output_schema=self._output_schema,
            graph_kwargs=dict(self._graph_kwargs),
            nodes=tuple(self._nodes),
            edges=tuple(self._edges),
            conditional_edges=tuple(self._conditional_edges),
            compile_options=dict(compile_options or {}),
            callbacks=tuple(callbacks),
            default_node_policy=self._default_node_policy,
        )


__all__ = [
    "AgentGraphSpec",
    "AgentNodeSpec",
    "CallableNodeSpec",
    "ConditionalEdgeSpec",
    "EdgeSpec",
    "GraphNodeSpec",
    "NodeGuardrailPolicy",
    "PlatformStateGraph",
    "coerce_node_guardrail_policy",
]
