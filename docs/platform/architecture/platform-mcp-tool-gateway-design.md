# Platform MCP Tool Gateway Implementation Design

Back to the [architecture index](index.md).

## Status

Final architecture decision for implementing the platform MCP tool gateway,
updated after independent architecture review.

The design is based on the current `bot_platform` repository and the
`C:\Projects\sysadmin_mcp_kit` MCP implementation. It assumes the platform will
eventually expose selected tools through a separate MCP service, while also
supporting an in-process wrapper path for any LangGraph/LangChain agent or
platform service during migration.

## Goals

- Create a central platform-owned tool execution layer.
- Allow selected existing Python/LangChain tools to be registered with minimal
  changes to their current implementation.
- Expose selected platform tools through a separate MCP service.
- Allow any platform consumer to use the same tool layer: LangGraph agents,
  `bot_service`, future agents, MCP clients, tool workers, and external MCP
  adapters.
- Support both internal Python tools and external MCP servers.
- Support per-tool policy for roles, scopes, side effects, sensitivity,
  approval, execution mode, queueing, timeouts, and guardrails.
- Preserve current agent-specific LangGraph state tools locally for now.
- Prepare a single place where the future tool guardrails layer can be enforced.

## Non-Goals

- Do not migrate LangGraph state mutation tools in the first phase.
- Do not expose OpenAI-compatible tool-calling at the `openai_proxy` boundary in
  the first phase.
- Do not build a generic MCPO/OpenAPI bridge unless a concrete non-MCP HTTP API
  must be exposed as MCP without writing a Python adapter.
- Do not move policy entirely into a database in the first phase.
- Do not replace existing agent guardrail middleware. The tool gateway adds a
  second enforcement boundary.
- Do not make Mycroft, DeepAgents, or any single current agent the architecture
  owner. Mycroft is only a pilot integration used to validate the generic
  LangChain/LangGraph adapter.

## Pilot Tool Registrations

These tools are pilot registrations chosen from current repository usage. They
are not Mycroft-specific architecture requirements. The platform registry must
not depend on Mycroft config, DeepAgents, subagents, or Mycroft runtime
semantics.

The first pilot migration batch should include these tools only.

| Tool | Source | Notes |
| --- | --- | --- |
| `kpi_staff_structure_fuzzy_search` | `agents.mycroft_agent.scenarios.kpi_agent.tools:build_kpi_staff_structure_fuzzy_search_tool` | Internal read-only SQLite-backed fuzzy lookup. |
| `search_marketing_materials` | `agents.gaz_agent.marketing_tools:build_marketing_document_tools` | Internal document retrieval through `GazDocumentsClient`. |
| `read_marketing_material` | `agents.gaz_agent.marketing_tools:build_marketing_document_tools` | Internal document read by candidate id. |
| `get_marketing_branch_pack` | `agents.gaz_agent.marketing_tools:build_marketing_document_tools` | Internal document pack retrieval. |
| `estimate_marketing_research_cost` | `agents.gaz_agent.marketing_tools:build_marketing_document_tools` | Internal cost/effort estimate. |
| `web_search` | `agents.mycroft_agent.web_search_subagent:build_web_search_tool` | External Yandex Search. Treat output as untrusted web content. Do not require the tool queue for first pilot migration; either leave the current subagent unchanged or migrate it as a synchronous `per_call` tool first. |
| `store_artifact_tool` | New platform adapter around `agents.store_artifacts.store_chapters` | File/artifact write side effect. Do not expose the current runtime-locale tool directly because it expects `ToolRuntime.state["locale"]`. |

Agent-specific state tools such as `commit_ideas`,
`commit_final_docset`, `commit_artifact_final_text`, and similar graph-state
commit tools remain local to their agents.

Implementation sequence:

- Phase 0/1 should validate KPI, marketing, and artifact storage through the
  generic in-process wrapper path first.
- `web_search` stays in target scope, but it should not force queue
  implementation. Keep the current subagent path until either synchronous
  `per_call` wrapping is proven safe or the tool queue exists.

## Key Design Decision

Build the platform tool layer in two parts:

```text
platform_tools core
  - registry
  - tool loading
  - LangChain wrappers
  - policy checks
  - guardrail hooks
  - queue integration
  - audit logging

services/platform_mcp service
  - FastMCP server
  - OAuth validation
  - MCP tool exposure
  - MCP progress and elicitation integration
  - delegates execution to platform_tools
```

This avoids a big-bang migration. Any agent or service that supports the generic
consumer contract can use `platform_tools` directly inside the current process
first. The separate MCP service can expose exactly the same registered tools
later.

## Platform Consumers

The platform tool gateway supports multiple consumers through adapters over the
same `platform_tools` core:

| Consumer | Adapter | Notes |
| --- | --- | --- |
| LangGraph/LangChain agents | `platform_tools.langchain` | Returns `BaseTool` wrappers usable by any agent. |
| `bot_service` agent initialization | `bot_service.platform_tooling` | Resolves configured platform tools for an agent and injects them only through an explicit supported init contract. |
| Platform MCP service | `services/platform_mcp` | Exposes selected registry tools to MCP clients. |
| Tool queue workers | `services/tool_queue` | Executes queued tool jobs using the same registry and policy. |
| External MCP adapters | `platform_tools.external_mcp` | Proxies selected external MCP tools through platform policy. |

No consumer owns tool policy. All consumers delegate to `platform_tools`.

## Generic Agent Integration Contract

Agents should not load platform tools through Mycroft-specific config. The
generic contract is:

- `platform_tools.langchain.build_langchain_platform_tools(...)` creates
  LangChain-compatible tools for any agent.
- Agents that support additive tool injection should accept
  `extra_tools: Sequence[Any] | None = None` or
  `platform_tools: Sequence[Any] | None = None`.
- Existing `tools` parameters must not be assumed additive because several
  agents use `tools` as a replacement for built-in tools.
- `bot_service` may inject platform tools only when an agent explicitly supports
  the configured injection parameter.
- Agent-local state mutation tools remain local unless separately adapted to the
  platform result/state-update contract.

Firm rule: `platform_tools` must not import from `agents.mycroft_agent.*` except
through configured pilot tool import paths. No platform module may depend on
Mycroft config classes, DeepAgents middleware, Mycroft subagent loaders, or
Mycroft-specific runtime state.

## Final Decisions From Architecture Review

These decisions settle the main design questions for the first implementation
cycle.

1. Keep the two-layer architecture:
   - `platform_tools` is the source of truth for tool loading, policy,
     execution, result normalization, and audit.
   - `services/platform_mcp` is a transport adapter over `platform_tools`, not a
     second implementation of tool policy.
2. Start with the generic in-process LangChain/LangGraph adapter before
   requiring a running MCP service. Mycroft may be the first pilot adopter, but
   the adapter must be agent-neutral.
3. Keep LangGraph state mutation tools local until the platform has an explicit
   state-update result contract.
4. Phase 0 supports synchronous execution only. Queue support is Phase 3. No
   first pilot tool may require queue infrastructure.
5. `web_search` is not a singleton. If migrated before the queue exists, it must
   use `lifecycle = "per_call"` because `YandexSearchTool` mutates instance
   state during `_run`.
6. Tool config must separate:
   - side effect: `none`, `read`, `write`;
   - network access: `none`, `internal`, `external`;
   - data boundary: `local`, `internal_service`, `external_service`;
   - sensitivity: `public`, `internal`, `confidential`, `regulated`.
7. OAuth scopes apply to MCP/external callers only. In-process agent calls use
   trusted platform source, `allowed_roles`, and optional `allowed_consumers`.
8. Internal service tokens authenticate the caller service only. Per-user
   authorization requires either delegated/OBO OAuth tokens or signed/gateway
   injected trusted headers accepted only from callers with
   `platform:tools:impersonate`.
9. Approval fails closed. If a tool requires approval and no approval provider is
   available, the call returns `blocked` or `approval_required`; it must not
   execute silently.
10. First MCP release may use static FastMCP tool registration and call-time
    policy enforcement. Tools with role-sensitive names/descriptions or
    confidential metadata should keep `expose.mcp = false` until filtered
    discovery is implemented.
11. External MCP servers are `public_untrusted` by default. Raw internal or
    confidential data may be forwarded only to `platform_trusted` servers unless
    a reviewed per-tool exception exists.
12. External MCP `stdio` servers are Phase 4 only. They require repo-configured
    command allowlists, explicit argv arrays, controlled working directories,
    environment allowlists, no inherited secrets by default, timeout, and output
    limits.
13. Queued jobs are owner-bound, TTL-limited, cancellable, and must not store raw
    bearer tokens or unredacted confidential outputs in Redis.
14. Tool schemas exposed to MCP must strip infrastructure parameters such as
    `runtime`, `config`, `ctx`, `callbacks`, `ToolRuntime`, and trusted context
    fields.
15. Synchronous tools called from async MCP handlers must run in a threadpool so
    the MCP event loop is not blocked.
16. Multi-tool builders need per-tool policy overrides because one builder can
    return tools with different sensitivity and exposure risk.

## Enforcement Order

Every platform tool call should follow the same order:

```text
resolve tool descriptor
  -> build trusted runtime context
  -> schema validate and coerce model-provided args
  -> policy authorization
  -> argument guardrails and privacy transforms
  -> approval if required
  -> execute sync / queued / external MCP
  -> result normalization
  -> result guardrails, redaction, source labeling, and minimization
  -> audit decision and outcome
  -> return formatted result
```

Policy checks that depend on argument values, such as `sensitive_args`, must run
after schema validation and before execution. Audit is allowed to record both
policy decisions and final outcomes, but must use redacted payloads.

## Existing Patterns To Reuse

- `bot_service/agent_registry.py`
  - Already inspects agent `initialize_agent(...)` signatures and forwards only
    supported initialization parameters.
- `agents/*/agent.py`
  - Several agents already accept optional tool lists or agent-specific init
    params. The platform integration should use explicit additive injection
    contracts, not implicit replacement.
- `agents/mycroft_agent/cli_config.py` and
  `agents/mycroft_agent/configured_agent.py`
  - Useful pilot examples for config-driven internal and MCP tool loading. They
    must not define the platform architecture.
- `agents/sysadmin_agent/mcp_utils.py`
  - Good MCP client/interceptor reference for auth, context, elicitation,
    progress callbacks, and remote MCP tool loading.
- `C:\Projects\sysadmin_mcp_kit`
  - Good server reference for FastMCP, streamable HTTP, OAuth introspection,
    progress, elicitation, result pagination, and audit patterns.
- `platform_guardrails/*`
  - Existing reusable guardrail context, scanner, privacy, middleware, and
    audit concepts.

## Architecture

```text
Agent or MCP client
        |
        v
LangChain platform wrapper OR FastMCP tool endpoint
        |
        v
platform_tools.context builds trusted runtime context
        |
        v
platform_tools.policy authorizes the call
        |
        v
platform_tools.guardrails scans/transforms args
        |
        v
platform_tools.invoker executes one of:
  - internal Python/LangChain tool
  - queued tool job
  - external MCP server tool
        |
        v
platform_tools.guardrails scans/minimizes result
        |
        v
platform_tools.audit records decision and outcome
        |
        v
Tool result returns to agent or MCP client
```

## Trust Boundaries

The model can choose a tool name and arguments, but it must not be trusted to
provide identity or authorization context.

Trusted context must come from:

- `RunnableConfig.configurable` for in-process agent calls.
- MCP access token claims for external MCP clients.
- Signed or gateway-injected internal HTTP headers only when the caller has the
  dedicated internal impersonation scope.
- Server-side config for tool metadata, sensitivity, side effects, and external
  server trust.

The following fields must never be trusted when supplied as model tool
arguments:

- `user_id`
- `user_role`
- `tenant_id`
- `thread_id`
- `conversation_id`
- `agent_id`
- `allowed_roles`
- `allowed_consumers`
- `required_mcp_scopes`
- `approval_required`

## Tool Policy Source

Use versioned TOML or YAML files in the repository for the first phase.

Recommended path:

```text
data/config/platform_tools/tools.toml
```

Rationale:

- Tool import paths, side effects, sensitivity, and execution mode are
  engineering/security concerns and should go through code review.
- Policy needs tests and deploy discipline because mistakes can expose data or
  enable unwanted side effects.
- Tool definitions should change rarely.
- Role allowlists and approval requirements may change more often, but should
  still be reviewed until an audited admin UI exists.

Later, split policy ownership:

| Policy Area | First Phase Owner | Expected Change Frequency | Later Storage |
| --- | --- | --- | --- |
| Tool import path and builder params | Platform engineering | Rare | Repository config |
| Side effect and sensitivity classification | Security/platform owner | Rare to occasional | Repository config |
| External server trust | Security/platform owner | Occasional | Repository config plus emergency disable |
| Role allowlists | Product/platform admin | Occasional | Database or policy service |
| Approval requirement | Product/security | Occasional | Database or policy service |
| Emergency disable | On-call/platform owner | Immediate when needed | Database/Redis override |

The first implementation can add a read-only file policy plus an optional
runtime override file or Redis flag for emergency tool disablement.

## Tool Config Shape

Example TOML:

```toml
[server]
config_version = 1
default_timeout_seconds = 30

[[tools]]
id = "kpi_staff_structure_fuzzy_search"
display_name = "KPI staff structure fuzzy search"
type = "internal_python"
import = "agents.mycroft_agent.scenarios.kpi_agent.tools:build_kpi_staff_structure_fuzzy_search_tool"
select = ["kpi_staff_structure_fuzzy_search"]
lifecycle = "singleton"

[tools.expose]
inprocess = true
mcp = true

[tools.params]
database_path = "data/kpi/kpi.sqlite"
default_limit_per_field = 8
default_min_score = 0.35

[tools.policy]
side_effect = "none"
network_access = "none"
data_boundary = "local"
sensitivity = "internal"
allowed_roles = ["default", "kpi_user", "admin"]
allowed_consumers = ["agent:pilot_kpi", "agent:kpi_bi"]
required_mcp_scopes = ["platform:tools"]
approval = "never"
external_access = false

[tools.execution]
mode = "sync"
timeout_seconds = 10
progress = "none"

[tools.guardrails]
scan_args = true
scan_result = true
anonymize_for_external = false
result_minimization = "default"

[[tools]]
id = "marketing_documents"
type = "internal_python"
import = "agents.gaz_agent.marketing_tools:build_marketing_document_tools"
select = [
  "search_marketing_materials",
  "read_marketing_material",
  "get_marketing_branch_pack",
  "estimate_marketing_research_cost",
]
lifecycle = "singleton"

[tools.expose]
inprocess = true
mcp = false

[tools.params]
locale = "ru"
docs_collection = "gaz"
timeout_seconds = 20.0

[tools.policy]
side_effect = "none"
network_access = "internal"
data_boundary = "internal_service"
sensitivity = "confidential"
allowed_roles = ["default", "sales_manager", "marketing_analyst", "admin"]
allowed_consumers = ["agent:pilot_sales", "agent:marketing_analyst"]
required_mcp_scopes = ["platform:tools"]
approval = "never"
external_access = false

[tools.execution]
mode = "sync"
timeout_seconds = 30
progress = "mcp"

[[tools.overrides]]
name = "estimate_marketing_research_cost"

[tools.overrides.policy]
sensitivity = "internal"

[[tools.overrides]]
name = "read_marketing_material"

[tools.overrides.policy]
sensitivity = "confidential"

[[tools]]
id = "web_search"
type = "internal_python"
import = "agents.mycroft_agent.web_search_subagent:build_web_search_tool"
select = ["web_search"]
lifecycle = "per_call"

[tools.expose]
inprocess = true
mcp = true

[tools.params]
max_results = 5
summarize = true

[tools.policy]
side_effect = "none"
network_access = "external"
data_boundary = "external_service"
sensitivity = "public"
allowed_roles = ["default", "sales_manager", "marketing_analyst", "admin"]
allowed_consumers = ["agent:pilot_sales", "agent:web_search"]
required_mcp_scopes = ["platform:tools"]
approval = "policy"
external_access = true

[tools.execution]
mode = "sync"
timeout_seconds = 90
progress = "mcp"

[[tools]]
id = "store_artifact_tool"
type = "platform_builtin"
builtin = "store_artifact_tool"
lifecycle = "stateless"

[tools.expose]
inprocess = true
mcp = true

[tools.policy]
side_effect = "write"
network_access = "internal"
data_boundary = "internal_service"
sensitivity = "internal"
allowed_roles = ["default", "sales_manager", "admin"]
allowed_consumers = ["agent:pilot_sales", "agent:artifact_creator"]
required_mcp_scopes = ["platform:tools"]
approval = "tool_config"
external_access = false

[tools.execution]
mode = "sync"
timeout_seconds = 30
progress = "none"
```

Config rules:

- `required_mcp_scopes` applies only when the caller enters through MCP or an
  external gateway path.
- In-process calls are authorized through `allowed_roles`, `allowed_consumers`, and
  trusted runtime source.
- `lifecycle = "singleton"` is allowed only for thread-safe tools. Use
  `per_call` for tools with mutable instance state and `stateless` for pure
  platform built-ins.
- Bundle-level policy is the default for all tools returned by a builder.
  `tools.overrides` can tighten or relax individual selected tools.
- A tool with `expose.mcp = false` may still be available to in-process agents
  through the same wrapper layer.

## Schema Strategy

Do not require significant changes to existing tools.

Schema resolution order:

1. If a builder returns LangChain tools, use each tool's existing `args_schema`.
2. If the tool is a plain callable, infer parameters from function signature and
   docstring, then create a Pydantic model.
3. If inference is incomplete or unsafe, require a sidecar schema in tool config.
4. Keep security metadata outside inferred schema. Introspection cannot know
   side effects, sensitivity, approval, or trust.
5. Strip infrastructure-only parameters from public schemas, including
   `runtime`, `config`, `ctx`, `callbacks`, `ToolRuntime`, `RunnableConfig`, and
   all trusted context fields.

This preserves current tool implementations while still giving the platform
explicit policy metadata.

## Result Strategy

Internally, every tool result should be normalized into:

```python
class PlatformToolResult(BaseModel):
    status: Literal["ok", "error", "blocked", "approval_required"]
    tool_name: str
    content: str | None = None
    structured_content: dict[str, Any] | list[Any] | None = None
    artifacts: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    user_message: str | None = None
```

LangChain compatibility wrapper:

- Returns plain text for current agents unless the original tool naturally
  returns dict-like content.
- Preserves current model expectations for `store_artifact_tool` by returning a
  Markdown link string.

MCP compatibility wrapper:

- Returns structured content when possible.
- Uses text content for simple string results.
- Does not include raw internal exceptions or secrets.

Result normalization must treat conventional error payloads as errors. For
example, marketing tools currently return dictionaries shaped like
`{"status": "error", "message": "..."}` when their backing service fails; the
platform result layer must convert those into sanitized `PlatformToolResult`
errors before the text reaches a model or MCP client.

## Runtime Context

Create a platform tool context that can map to guardrail context.

```python
class PlatformToolContext(BaseModel):
    request_id: str
    tenant_id: str | None = None
    actor_client_id: str | None = None
    consumer_type: Literal["agent", "service", "mcp_client", "tool_worker"]
    consumer_id: str
    user_id: str | None = None
    user_role: str = "default"
    thread_id: str | None = None
    conversation_id: str | None = None
    agent_id: str | None = None
    scopes: list[str] = []
    source: Literal["langchain", "mcp", "tool_worker", "service"]
    tool_name: str | None = None
    mcp_session_id: str | None = None
    metadata: dict[str, Any] = {}
```

## Execution Modes

Per-tool execution mode:

| Mode | Behavior | First Use |
| --- | --- | --- |
| `sync` | Execute in the current process and return final result. | KPI search, marketing reads, artifact storage. |
| `queued_wait` | Enqueue a tool job, wait for completion, stream/poll progress, return final result. | Future web search or expensive research after the tool queue exists. |
| `queued_detached` | Enqueue and return job id immediately. Caller uses polling tools. | Later for long-running workflows. |
| `external_mcp` | Forward to a configured external MCP server after policy and guardrails. | Later mixed external server integration. |

For MCP calls, `queued_wait` should use `ctx.report_progress(...)` while waiting.
For in-process LangGraph calls, `queued_wait` should publish custom stream events
when a LangGraph stream writer is available.

Phase 0 and Phase 1 implement `sync` only. `queued_wait` and
`queued_detached` are designed now but implemented only in Phase 3. If a
descriptor is executed by the future tool worker with
`context.source == "tool_worker"`, the invoker must force direct execution and
must not recursively enqueue the same tool.

## Approval Modes

Per-tool approval mode:

| Mode | Behavior |
| --- | --- |
| `never` | No approval required after policy allows the call. |
| `always` | Ask approval for every call. |
| `policy` | Ask only when policy classifies the call as risky. |
| `sensitive_args` | Ask when regex/schema rules match sensitive arguments. |
| `tool_config` | Use tool-specific approval config. |

MCP approval implementation:

- Use MCP elicitation through `ctx.elicit(...)`.
- Follow the sysadmin kit pattern, but use redacted prompts and do not log
  secrets.

In-process agent implementation:

- Consumer-specific approval hooks, such as DeepAgents `interrupt_on`, may be
  used by that consumer's adapter, but they are not part of the platform core.
- For tools not covered by a consumer-specific approval hook, the platform
  wrapper can later use LangGraph `interrupt(...)` if runtime is available.
- First phase may configure `approval=never` only for tools explicitly
  classified as safe. If policy says approval is required and no approval
  provider is available, the call must fail closed with `blocked` or
  `approval_required`.

## Progress Flow

MCP progress:

```text
MCP client calls tool
  -> service validates auth
  -> policy/guardrails allow
  -> invoker starts sync or queued execution
  -> invoker calls ProgressSink.report(...)
  -> MCP sink maps to ctx.report_progress(...)
  -> final result returned
```

LangGraph progress:

```text
Agent calls platform wrapper
  -> wrapper builds context
  -> invoker starts queued_wait
  -> wrapper polls job events
  -> if get_stream_writer() is available, emits:
       {"type": "tool_progress", "tool": "...", "progress": 0.4, "message": "..."}
  -> final result returned
```

Progress messages and elicitation prompts are part of the security boundary.
They must be redacted with the same rules as audit logs and must not include raw
secrets, bearer tokens, Palimpsest mappings, or unminimized confidential tool
outputs.

## Queued Tool Flow

Use a new tool queue, not the existing agent task queue directly. The current
`services/task_queue` payload is conversation/agent specific. The tool queue can
reuse the Redis patterns but should have its own models and keys.

```text
Caller invokes queued tool
  -> platform_tools.invoker creates ToolJobPayload
  -> services.tool_queue.redis_queue stores job and emits queued event
  -> services.tool_queue.worker pops job
  -> worker loads platform tool registry
  -> worker calls PlatformToolInvoker in source="tool_worker" mode
  -> worker publishes progress/chunk/result/failure
  -> caller waits or returns job id depending on mode
```

Required event types:

- `queued`
- `running`
- `progress`
- `completed`
- `failed`
- `blocked`
- `approval_required`

Queue safety rules:

- Jobs are bound to an owner and subject context.
- Jobs have TTLs and heartbeat/watchdog cleanup.
- Jobs can be cancelled.
- Write or external tools should support idempotency keys where duplicate
  execution could create duplicate side effects or duplicated external requests.
- Redis payloads must not contain raw bearer tokens, OAuth refresh tokens,
  Palimpsest mappings, or unredacted confidential outputs.

## External MCP Server Flow

External MCP servers are `public_untrusted` by default. The platform supports
three trust tiers:

| Tier | Meaning | Raw internal/confidential data |
| --- | --- | --- |
| `platform_trusted` | Operated inside the trusted platform environment. | Allowed when tool policy permits. |
| `partner_restricted` | Contracted or semi-trusted partner endpoint. | Blocked by default; reviewed per-tool exception required. |
| `public_untrusted` | Public or unknown external MCP server. | Blocked. Only public or anonymized/minimized data may be sent. |

```text
Model requests platform-exposed external tool
  -> platform policy checks tool and server
  -> guardrails scan model-generated args
  -> privacy policy decides raw, anonymized, or blocked forwarding
  -> external MCP client calls remote server
  -> remote result is source-labeled as external/untrusted
  -> tool-output scanner checks prompt injection, secrets, malicious URLs, etc.
  -> result minimizer trims large output
  -> sanitized result returns to model
```

Controls:

- Prefix external tools by server name unless explicitly disabled.
- Block tool name collisions.
- Keep per-server trust metadata.
- Do not forward platform context fields as normal tool args.
- Do not forward raw PII to untrusted external servers.
- Scan external tool output before it re-enters the model.

`stdio` external MCP servers are Phase 4 only. They can launch local processes
with platform service privileges, so they require stricter rules than HTTP/SSE:

- command must be selected from a repo-configured allowlist;
- no caller-supplied shell strings;
- explicit argv array only;
- controlled working directory;
- environment allowlist, with no inherited secrets by default;
- timeout and maximum output size;
- audit of command id, not raw secrets.

## Hidden Threats From MCP Exposure

MCP exposes tool names, descriptions, and schemas as prompt context. Treat those
descriptions as security-relevant text.

Risks and controls:

| Risk | Control |
| --- | --- |
| Tool name collision or shadowing | Registry rejects duplicate exposed names unless explicitly namespaced. |
| Model spoofs identity in args | Trusted identity comes from runtime context only. |
| Data exfiltration to external MCP server | Per-server trust policy plus privacy transform before forwarding. |
| Tool output prompt injection | Scan tool output and source-label external content. |
| Side effects without review | Per-tool approval policy. |
| Sensitive data in errors/progress | Redact errors, progress, audit logs. |
| Capability discovery leaks internal systems | Expose only active tools allowed for the caller scope/role. |
| Queued job replay | Use job ids, owner binding, TTL, idempotency keys where needed. |
| Long output overloads model context | Pagination, summarization, and result minimization. |

## Modules To Create

Phase 0 should create only the core in-process surface:

- `platform_tools/config.py`
- `platform_tools/selection.py`
- `platform_tools/registry.py`
- `platform_tools/descriptors.py`
- `platform_tools/invoker.py`
- `platform_tools/langchain.py`
- `platform_tools/results.py`
- `platform_tools/policy.py`
- `platform_tools/audit.py`
- `platform_tools/builtins/store_artifact.py`

The approval, progress, external MCP, MCP service, and queue modules are still
specified below, but they are implemented in later phases unless a pilot
tool strictly requires them.

### `platform_tools/__init__.py`

Exports public APIs:

- `load_platform_tool_registry`
- `build_langchain_platform_tools`
- `PlatformToolInvoker`
- `PlatformToolContext`
- `PlatformToolResult`

### `platform_tools/config.py`

Responsibilities:

- Load and validate platform tool config.
- Expand environment variable placeholders.
- Resolve relative paths against repo root.
- Validate unique tool ids and exposed tool names.

Functions/classes:

- `class PlatformToolsSettings(BaseModel)`
  - Root config model.
- `class ToolSpec(BaseModel)`
  - One configured tool or tool bundle.
- `class ToolExposeSpec(BaseModel)`
  - In-process and MCP exposure flags.
- `class ToolOverrideSpec(BaseModel)`
  - Per-tool overrides for tools returned by a multi-tool builder.
- `class ToolPolicySpec(BaseModel)`
  - Role/scope/side-effect/sensitivity/approval policy.
- `class ToolExecutionSpec(BaseModel)`
  - Sync/queued/external execution settings.
- `class ToolGuardrailSpec(BaseModel)`
  - Guardrail flags for args/results/privacy.
- `class ExternalMCPServerSpec(BaseModel)`
  - External MCP server connection and trust settings.
- `load_platform_tools_config(path: str | Path | None = None) -> PlatformToolsSettings`
  - Loads TOML/YAML/JSON and validates it.
- `resolve_platform_tools_config_path(raw_path: str | Path | None) -> Path`
  - Uses env fallback such as `PLATFORM_TOOLS_CONFIG`.
- `expand_env_placeholders(value: Any, *, owner: str) -> Any`
  - Replaces `${ENV_NAME}` in config values.
- `validate_unique_exposed_names(settings: PlatformToolsSettings) -> None`
  - Fails fast on exposed name conflicts.

### `platform_tools/selection.py`

Responsibilities:

- Define the generic consumer-side tool selection contract.
- Keep consumer config parsing separate from any specific agent's local config
  format.

Functions/classes:

- `class PlatformToolSelection(BaseModel)`
  - Selected platform tool ids, optional config path, and optional consumer
    metadata.
- `load_platform_tool_selection(raw: Any) -> PlatformToolSelection`
  - Parses a neutral selection object.
- `normalize_selected_tool_ids(selection: PlatformToolSelection) -> tuple[str, ...]`
  - Produces stable selected ids for registry lookup.

### `platform_tools/context.py`

Responsibilities:

- Build trusted runtime context for policy and guardrails.
- Convert platform context to `GuardrailContext`.

Functions/classes:

- `class PlatformToolContext(BaseModel)`
  - Trusted context model described above.
- `context_from_runnable_config(config: RunnableConfig | None, state: Mapping[str, Any] | None, *, tool_name: str) -> PlatformToolContext`
  - Builds context for in-process LangChain calls.
- `context_from_mcp(ctx: Context, *, tool_name: str, trusted_headers: Mapping[str, str] | None = None) -> PlatformToolContext`
  - Builds context for MCP calls from token/session/request metadata.
- `context_from_tool_job(payload: ToolJobPayload) -> PlatformToolContext`
  - Rehydrates context for queued worker execution.
- `to_guardrail_context(context: PlatformToolContext) -> GuardrailContext`
  - Bridges into existing `platform_guardrails`.
- `trusted_context_overlay(context: PlatformToolContext, headers: Mapping[str, str], *, allowed_scopes: set[str]) -> PlatformToolContext`
  - Applies internal headers only if caller has a trusted internal scope.

### `platform_tools/descriptors.py`

Responsibilities:

- Represent loaded tools independent of LangChain or MCP.
- Preserve original tool metadata and schema.

Functions/classes:

- `class PlatformToolDescriptor(BaseModel)`
  - Tool id, exposed name, description, schema, policy, execution, source,
    lifecycle, and factory reference.
- `class LoadedToolBundle(BaseModel)`
  - A builder result containing one or more descriptors.
- `descriptor_from_langchain_tool(tool: BaseTool, spec: ToolSpec) -> PlatformToolDescriptor`
  - Extracts name, description, args schema, and return behavior.
- `descriptor_for_builtin_tool(spec: ToolSpec) -> PlatformToolDescriptor`
  - Creates descriptor for platform built-ins such as artifact storage.
- `mcp_annotations_for_descriptor(descriptor: PlatformToolDescriptor) -> ToolAnnotations`
  - Maps side effects to MCP annotations.

### `platform_tools/registry.py`

Responsibilities:

- Import Python tool builders.
- Build selected tools from bundles.
- Create the registry used by wrappers and MCP service.
- Store tool factories and lifecycle mode so non-thread-safe tools can be built
  per call.

Functions/classes:

- `class PlatformToolRegistry`
  - Holds descriptors by exposed name and id.
- `load_platform_tool_registry(settings: PlatformToolsSettings) -> PlatformToolRegistry`
  - Main registry loading entrypoint.
- `load_internal_tool_bundle(spec: ToolSpec) -> list[Any]`
  - Imports `module:function`, calls builder with params, normalizes list/singleton.
- `select_tools_from_bundle(tools: list[Any], selected_names: list[str] | None) -> list[Any]`
  - Selects named tools from a builder result.
- `load_builtin_tool(spec: ToolSpec) -> Any`
  - Returns platform-provided tool implementation.
- `load_external_mcp_tool_descriptors(settings: PlatformToolsSettings) -> list[PlatformToolDescriptor]`
  - Creates descriptors for configured external MCP tools without exposing them
    directly to agents yet.

### `platform_tools/builtins/store_artifact.py`

Responsibilities:

- Hold platform-owned tool implementations that should not depend on
  LangChain `ToolRuntime`.

Functions:

- `build_store_artifact_tool() -> BaseTool`
  - Builds a `store_artifact_tool` compatible tool.
- `store_artifact(title: str, artifact: str, locale: str = "en") -> PlatformToolResult`
  - Calls `agents.store_artifacts.store_chapters`, returns stable structured
    data and Markdown link.

### `platform_tools/policy.py`

Responsibilities:

- Enforce per-tool authorization and execution policy.

Functions/classes:

- `class ToolPolicyDecision(BaseModel)`
  - `allowed`, `action`, `reason`, `requires_approval`, `categories`,
    `metadata`.
- `class ToolPolicyEngine`
  - Evaluates tool calls.
- `ToolPolicyEngine.evaluate(descriptor, args, context) -> ToolPolicyDecision`
  - Checks active flag, role, `allowed_consumers`, MCP scopes when
    `context.source == "mcp"`, side effects, external access, and emergency
    disable.
- `ToolPolicyEngine.requires_approval(descriptor, args, context) -> bool`
  - Applies approval mode.
- `ToolPolicyEngine.validate_external_server_allowed(descriptor, context) -> ToolPolicyDecision`
  - Checks external MCP server trust and access.
- `load_policy_overrides(...)`
  - Optional first-phase hook for emergency disables.

### `platform_tools/guardrails.py`

Responsibilities:

- Provide the future tool guardrail integration point.
- In the first phase, implement no-op/default scanning hooks with audit
  decisions.

Functions/classes:

- `class ToolGuardrailEngine`
  - Applies argument and result guardrails.
- `prepare_arguments(descriptor, args, context) -> tuple[dict[str, Any], list[GuardrailDecision]]`
  - Schema-normalizes and scans tool args.
- `prepare_external_arguments(descriptor, args, context) -> tuple[dict[str, Any], list[GuardrailDecision]]`
  - Applies privacy rules before external MCP forwarding.
- `process_result(descriptor, raw_result, context) -> PlatformToolResult`
  - Scans, redacts, anonymizes, labels, and minimizes result.
- `sanitize_error(descriptor, exc, context) -> PlatformToolResult`
  - Converts exceptions into safe user-visible errors.

### `platform_tools/progress.py`

Responsibilities:

- Abstract progress reporting across MCP, LangGraph, and queue workers.

Functions/classes:

- `class ProgressSink`
  - Interface with `report(progress: float, message: str, total: float | None = None)`.
- `class NoopProgressSink`
  - No-op implementation.
- `class MCPProgressSink`
  - Calls `ctx.report_progress`.
- `class LangGraphProgressSink`
  - Uses `get_stream_writer()` when available.
- `class QueueProgressSink`
  - Publishes progress events to Redis.

### `platform_tools/approval.py`

Responsibilities:

- Abstract approval collection across MCP and in-process execution.

Functions/classes:

- `class ApprovalRequest(BaseModel)`
  - Tool name, summary, args preview, risk reason.
- `class ApprovalResponse(BaseModel)`
  - Approved/rejected/edited decision.
- `class ApprovalProvider`
  - Interface.
- `class MCPApprovalProvider`
  - Uses MCP elicitation.
- `class NoopApprovalProvider`
  - Allows only when policy says approval is not required.
- `class LangGraphApprovalProvider`
  - Future implementation using LangGraph interrupts.
- `redacted_args_preview(args, descriptor) -> dict[str, Any]`
  - Builds safe approval prompt details.

### `platform_tools/results.py`

Responsibilities:

- Normalize tool return values.

Functions/classes:

- `class PlatformToolResult(BaseModel)`
  - Common result model.
- `normalize_raw_result(raw: Any, *, descriptor: PlatformToolDescriptor) -> PlatformToolResult`
  - Converts strings, dicts, Pydantic models, ToolMessages, and MCP results.
- `format_for_langchain(result: PlatformToolResult, descriptor) -> Any`
  - Preserves existing agent expectations.
- `format_for_mcp(result: PlatformToolResult, descriptor) -> Any`
  - Produces MCP-compatible return values.

### `platform_tools/invoker.py`

Responsibilities:

- Execute tools using policy, guardrails, approval, queueing, and audit.

Functions/classes:

- `class PlatformToolInvoker`
  - Main execution facade.
- `PlatformToolInvoker.invoke(tool_name, args, context, *, progress=None, approval=None) -> PlatformToolResult`
  - Synchronous main entrypoint.
- `PlatformToolInvoker.ainvoke(...) -> PlatformToolResult`
  - Async main entrypoint.
- `_invoke_internal(...)`
  - Calls internal Python/LangChain tool.
- `_invoke_queued_wait(...)`
  - Enqueues tool job and waits for completion.
- `_invoke_queued_detached(...)`
  - Enqueues and returns job id.
- `_invoke_external_mcp(...)`
  - Calls configured external MCP server.
- `_call_langchain_tool(...)`
  - Handles `invoke`/`ainvoke` differences and config propagation.

### `platform_tools/langchain.py`

Responsibilities:

- Build LangChain `BaseTool` wrappers for any platform tool descriptor.
- Preserve original names, descriptions, and argument schemas.
- Convert `RunnableConfig` and optional runtime state into
  `PlatformToolContext`.

Functions:

- `build_langchain_platform_tools(registry, selection, *, consumer) -> list[BaseTool]`
  - Builds wrappers for selected descriptors for a specific consumer.
- `make_langchain_tool(descriptor, invoker, *, consumer) -> BaseTool`
  - Creates a LangChain tool with the original name, description, and argument
    schema.
- `context_from_langchain_config(config, *, agent_id, source) -> PlatformToolContext`
  - Builds trusted platform context from LangChain runtime configuration.
- `invoke_from_langchain_tool(descriptor, args, runtime, config) -> Any`
  - Builds context, invokes the platform tool, and formats the result.

### `platform_tools/audit.py`

Responsibilities:

- Write safe structured audit logs.

Functions/classes:

- `class ToolAuditLogger`
  - JSONL or standard logging audit sink.
- `log_tool_call_started(...)`
- `log_tool_call_decision(...)`
- `log_tool_call_completed(...)`
- `log_tool_call_failed(...)`
- `redact_audit_payload(payload: Mapping[str, Any]) -> dict[str, Any]`

Audit records must never include raw secrets, bearer tokens, Palimpsest mappings,
or raw confidential tool outputs.

### `platform_tools/external_mcp.py`

Responsibilities:

- Manage external MCP clients and tool calls.

Functions/classes:

- `class ExternalMCPClientManager`
  - Holds `MultiServerMCPClient` instances or sessions.
- `load_external_mcp_clients(settings) -> ExternalMCPClientManager`
  - Builds clients from config.
- `call_external_mcp_tool(descriptor, args, context, progress) -> PlatformToolResult`
  - Calls external MCP tool after platform checks.
- `select_external_tool(server_name, tool_name, loaded_tools) -> Any`
  - Reuses generic tool selection and filtering logic.

### `bot_service/platform_tooling.py`

Responsibilities:

- Resolve platform tool selections from agent config.
- Build tools for agents that explicitly support platform tool injection.
- Avoid Mycroft-specific config paths and consumer-specific runtime assumptions.

Functions/classes:

- `class AgentPlatformToolSelection(BaseModel)`
  - Agent-level selection of registered platform tool ids.
- `load_agent_platform_tool_selection(raw) -> AgentPlatformToolSelection`
  - Parses optional `platform_tools` from an agent definition.
- `build_platform_tools_for_agent(agent_id, selection, runtime_context) -> list[Any]`
  - Builds LangChain-compatible wrappers through the generic adapter.
- `inject_platform_tools_if_supported(definition, params, tools) -> dict[str, Any]`
  - Injects tools only through an explicit supported init parameter.

## Services To Create

### `services/platform_mcp/config.py`

Responsibilities:

- Load MCP service settings.
- Load auth settings.
- Locate platform tools config.

Functions/classes:

- `class PlatformMCPServerSettings(BaseModel)`
- `class PlatformMCPOAuthSettings(BaseModel)`
- `class PlatformMCPSettings(BaseModel)`
- `load_settings(path: str | Path | None = None) -> PlatformMCPSettings`
- `to_auth_settings(...) -> AuthSettings`

### `services/platform_mcp/auth.py`

Responsibilities:

- Validate OAuth bearer tokens.
- Follow sysadmin kit introspection pattern without debug token logging.

Functions/classes:

- `class IntrospectionTokenVerifier(TokenVerifier)`
  - Calls configured introspection endpoint.
  - Validates active flag, issuer, resource/audience, required scopes.
  - Returns `AccessToken`.
- `extract_scopes(payload: dict[str, Any]) -> list[str]`
- `validate_resource(payload: dict[str, Any], expected_resource: str) -> bool`

### `services/platform_mcp/server.py`

Responsibilities:

- Build FastMCP server.
- Register platform tools dynamically.
- Build context and delegate calls to `PlatformToolInvoker`.

Functions/classes:

- `build_server(settings: PlatformMCPSettings) -> FastMCP`
  - Main service factory.
- `register_platform_tools(server: FastMCP, registry: PlatformToolRegistry, invoker: PlatformToolInvoker) -> None`
  - Registers one MCP tool per descriptor.
- `make_mcp_tool_handler(descriptor, invoker)`
  - Creates async tool handler for dynamic registration.
- `build_context_for_call(ctx: Context, descriptor) -> PlatformToolContext`
  - Uses token, session, request id, and trusted headers.
- `owner_id(ctx: Context) -> str`
  - Returns token client id or anonymous fallback.
- `transport_session_id(ctx: Context) -> str | None`
  - Mirrors sysadmin kit session extraction.

### `services/platform_mcp/main.py`

Responsibilities:

- Entrypoint for running the service.

Functions:

- `main() -> None`
  - Loads settings, builds server, runs `streamable-http`.

### `services/tool_queue/models.py`

Responsibilities:

- Define queued tool job contracts.

Functions/classes:

- `class ToolJobPayload(BaseModel)`
- `class ToolQueueEvent(BaseModel)`
- `class ToolJobStatus(str, Enum)`
- `class ToolJobResult(BaseModel)`

### `services/tool_queue/config.py`

Responsibilities:

- Define Redis keys, TTLs, heartbeat, watchdog timings, and worker limits.

Functions/classes:

- `class ToolQueueSettings(BaseSettings)`
- `settings`

### `services/tool_queue/redis_queue.py`

Responsibilities:

- Redis queue and event stream for tool jobs.

Functions/classes:

- `class RedisToolQueue`
- `enqueue(payload: ToolJobPayload) -> str`
- `mark_status(job_id, status, extra=None) -> None`
- `publish_event(event: ToolQueueEvent) -> None`
- `iter_events(job_id, include_status_snapshot=True)`
- `wait_for_completion(job_id, timeout)`
- `pop_job(timeout)`
- `store_result(job_id, result)`
- `store_failure(job_id, error)`
- `fail_stale_jobs(...)`

### `services/tool_queue/worker.py`

Responsibilities:

- Execute queued tool jobs outside the MCP request process.

Functions/classes:

- `class ToolWorker`
- `run() -> None`
- `worker_loop()`
- `_process_job(payload: ToolJobPayload)`
- `_heartbeat_loop(job_id)`

The worker must load the same platform tool registry as the MCP service.

## Modules To Change

### `pyproject.toml`

Change:

- Add a direct dependency on `mcp>=1.26.0,<2` if not already guaranteed by
  installed transitive dependencies.

Reason:

- The platform MCP service should depend explicitly on the MCP SDK because it
  imports FastMCP and auth classes directly.

### `bot_service/agent_registry.py`

Change:

- Parse optional `platform_tools` from each agent entry in
  `data/config/bot_service/load.json`.
- Build selected platform tools through `bot_service.platform_tooling`.
- Inject platform tools only when the target `initialize_agent(...)` signature
  has `extra_tools`, `platform_tools`, or an explicitly configured injection
  parameter.
- Never silently replace an agent's existing `tools` parameter. Some agents use
  `tools` as a replacement for built-ins rather than as an additive list.
- Preserve current behavior for agents without `platform_tools`.

### Agent `initialize_agent(...)` Contracts

Change per agent, only when that agent is intentionally adopted:

- Add an additive platform-tool parameter such as
  `extra_tools: Sequence[Any] | None = None` or
  `platform_tools: Sequence[Any] | None = None`.
- Merge the injected tools with the agent's built-in tools in a local,
  agent-owned order.
- Keep agent-local state mutation tools local unless they are explicitly adapted
  to the platform state-update contract.

### `docs/platform/architecture/index.md`

Change:

- Add link to this design document.

### `docker-compose.yml`

Change in service phase:

- Add optional `platform_mcp` service.
- Add optional `tool_worker` service when queued tools are enabled.

### `README.md`

Change:

- Add run commands for:
  - platform MCP service
  - tool queue worker
  - sample MCP client call

## Initial Built-In Store Artifact Adapter

Do not expose `agents.tools.store.store_artifact_tool` directly because it
depends on `ToolRuntime.state["locale"]`.

Create a platform built-in:

```python
def store_artifact(
    title: str,
    artifact: str,
    locale: str = "en",
) -> PlatformToolResult:
    ...
```

Behavior:

- Calls `agents.store_artifacts.store_chapters`.
- Returns:
  - `url`
  - `markdown_link`
  - `title`
  - `status`
- LangChain formatter returns the Markdown link string for compatibility.
- MCP formatter returns structured content plus text content.

## Generic In-Process Agent Flow

```text
Agent or bot_service selects platform tools
  -> platform_tools loads registry and descriptors
  -> platform_tools.langchain builds BaseTool wrappers
  -> agent receives wrappers through explicit additive injection
  -> model calls platform-wrapped tool
  -> wrapper builds PlatformToolContext from trusted config/runtime
  -> invoker runs policy, guardrails, approval, execution, result normalization, audit
  -> wrapper formats result for LangChain
  -> agent receives sanitized tool result
```

This path does not require the separate MCP service to be running.

## Platform MCP Service Flow

```text
MCP client connects to /mcp
  -> FastMCP validates bearer token through introspection
  -> client lists tools
  -> service returns only registered MCP-exposed tools
  -> client calls tool
  -> handler builds PlatformToolContext from token/session/request
  -> policy validates role/scope/tool access
  -> approval flow runs if required
  -> guardrail hooks process args
  -> invoker executes sync, queued, or external MCP
  -> progress is reported through ctx.report_progress
  -> guardrail hooks process result
  -> safe result returned
  -> audit event is written
```

## Auth Design

Use the sysadmin kit pattern:

- MCP server receives bearer token.
- Server introspects token through configured OAuth introspection endpoint.
- Server validates:
  - active token
  - issuer
  - resource/audience
  - required scope
  - client id
- Server never logs raw token or full introspection payload.

Recommended first scope:

```text
platform:tools
```

Optional future scopes:

```text
platform:tools:read
platform:tools:write
platform:tools:external
platform:tools:admin
```

Impersonation and delegated user context:

- A service credential identifies only the service client.
- Per-user authorization requires either a delegated/OBO user token or trusted
  subject headers injected by a gateway.
- Trusted subject headers are accepted only from callers with
  `platform:tools:impersonate`.
- Audit records must include both `actor_client_id` and subject `user_id` when
  they differ.

## Tool Discovery Rules

First MCP release uses static FastMCP registration for tools where
`expose.mcp = true`. Call-time policy enforcement is mandatory for every tool
call.

Static discovery means tool names, descriptions, and schemas can be visible to
any authenticated MCP caller that can list the server's tools. Therefore:

- Do not set `expose.mcp = true` for tools whose names, descriptions, or
  schemas reveal role-sensitive or confidential metadata.
- Do not expose confidential marketing or internal data tools over MCP until
  either their descriptions are safe for all authenticated callers or filtered
  discovery is implemented.
- Policy still checks active flag, role, scopes, external server trust, and
  emergency disable at call time.

Filtered per-caller discovery is a later enhancement unless confidential tools
must be MCP-visible before then.

## Testing Plan

Unit tests:

- Config loading and validation.
- Env placeholder expansion.
- Duplicate tool name rejection.
- LangChain tool schema extraction.
- Store artifact adapter formatting.
- Policy allow/block/approval decisions.
- Context extraction from RunnableConfig.
- Context extraction from mocked MCP context.
- Audit redaction.
- Result normalization.
- Queued job model serialization.

Phase 0/1 integration tests:

- Generic LangChain wrappers work outside Mycroft.
- At least one configured platform agent receives platform tools through an
  explicit additive injection contract.
- Pilot KPI, marketing, and artifact-storage tools run without a platform MCP
  service.

First MCP release integration tests:

- Platform MCP service lists configured tools.
- Platform MCP service rejects missing/invalid token.
- Platform MCP service rejects missing required scope.
- Platform MCP service calls KPI fuzzy search successfully.

Phase 3 queue integration tests:

- Platform MCP service reports progress for queued `web_search`.
- Tool worker executes queued job and publishes completion.

Phase 4 external MCP integration tests:

- External MCP forwarding blocks untrusted raw confidential input.

Regression tests:

- Existing agent configs still work when `platform_tools` is absent.
- Existing direct/internal tool paths remain unchanged.
- Agent-specific commit tools remain local.

## Implementation Phases

### Phase 0: Platform Core And Generic LangChain Adapter

- Create `platform_tools` package.
- Implement config, descriptors, registry, policy skeleton, result
  normalization, and audit skeleton.
- Implement the generic LangChain wrapper adapter.
- Implement built-in `store_artifact_tool`.
- Add pilot tool registrations.
- Add tests using at least one non-Mycroft dummy tool or dummy agent fixture.

Deliverable:

- Any agent can receive selected platform tools in-process through the generic
  adapter.

### Phase 1: First Pilot Agent Integrations

- Add optional `bot_service` agent config support for platform tool selection.
- Integrate one or more pilot agents explicitly.
- Mycroft may be one pilot, but it is not the architecture center and should
  not be the only acceptance target if a simpler agent can validate generic
  injection.
- Preserve existing direct internal tool paths during migration.

Deliverable:

- At least one configured platform agent uses platform tools through the generic
  adapter without requiring MCP.

### Phase 2: Separate MCP Service

- Create `services/platform_mcp`.
- Add FastMCP service with OAuth introspection.
- Register configured `expose.mcp = true` platform tools at startup.
- Support sync execution and MCP progress sink.
- Add run docs and docker-compose service.

Deliverable:

- External MCP clients can discover and call selected platform tools.

### Phase 3: Tool Queue

- Create `services/tool_queue`.
- Implement Redis queue, events, worker, watchdog.
- Add `queued_wait` execution mode.
- Configure `web_search` as queued if desired.

Deliverable:

- Long-running tools can execute outside MCP request process while preserving
  progress.

### Phase 4: External MCP Gateway

- Add `platform_tools/external_mcp.py`.
- Add external MCP server config.
- Add forwarding through `MultiServerMCPClient`.
- Enforce server trust and privacy policy before forwarding.

Deliverable:

- Platform can expose selected external MCP tools through the same policy and
  audit layer.

### Phase 5: Full Tool Guardrails

- Connect `platform_tools.guardrails` to `platform_guardrails` scanner and
  privacy rails.
- Add per-tool scanner profiles.
- Add output/source labeling and result minimization.
- Add policy-as-code or database override for role/approval changes.

Deliverable:

- Tool calls are centrally guarded, audited, and policy-controlled.

## Appendix: Mycroft Pilot Integration

This appendix describes one pilot adopter. It is not the platform integration
model. The platform model is the generic consumer adapter contract above.

Mycroft is useful as a pilot because it already demonstrates several current
tool-loading patterns:

- internal Python tool builders;
- external MCP server tools;
- DeepAgents subagents;
- tool lists passed into an agent factory.

Rules for the Mycroft pilot:

- Mycroft's config may select registered platform tool ids, but it must not
  define the platform tool registry schema.
- `agents.mycroft_agent.*` may call generic platform APIs, but
  `platform_tools.*` must not import Mycroft config classes, DeepAgents
  middleware, Mycroft subagent loaders, or Mycroft runtime state.
- Existing Mycroft `internal_tools` and `mcp` config remain backward compatible.
- Selected tools can move from Mycroft-local `internal_tools` into
  `platform_tools` gradually.

Pilot changes:

### `agents/mycroft_agent/cli_config.py`

- Parse an optional Mycroft-local `platform_tools` selection block.
- Translate that block into the generic `PlatformToolSelection` contract.
- Keep existing `internal_tools` and external `mcp` parsing unchanged.

Example:

```json
{
  "platform_tools": {
    "config_path": "data/config/platform_tools/tools.toml",
    "tools": [
      "store_artifact_tool",
      "kpi_staff_structure_fuzzy_search"
    ]
  }
}
```

### `agents/mycroft_agent/configured_agent.py`

- Call the generic platform loader to build LangChain wrappers.
- Append those wrappers to Mycroft's resolved tool list only as an explicit
  pilot behavior.
- Preserve current behavior when `platform_tools` is absent.

Suggested pilot order:

```python
internal_tools = build_internal_tools(mycroft_config.internal_tools)
platform_tools = build_langchain_platform_tools(...)
mcp_tools = asyncio.run(load_mcp_tools_from_config(mycroft_config.mcp))
tools = [*internal_tools, *platform_tools, *mcp_tools]
```

### `agents/mycroft_agent/subagent_loader.py`

- No platform requirement in Phase 0.
- Later, allow `web_search_agent` to receive a platform-provided `web_search`
  tool if the pilot chooses to migrate that path.

### `agents/mycroft_agent/web_search_subagent.py`

- Optional later change: accept `tools: list[Any] | None = None`.
- If tools are supplied, use those instead of constructing a local
  `YandexSearchTool`.
- This is not required for the first generic platform implementation.

### Mycroft JSON Configs

- The KPI pilot config may replace the direct internal import with a platform
  tool selection after the generic wrapper is implemented.
- The general Mycroft config may move `store_artifact_tool` to `platform_tools`
  after the built-in adapter is ready.
- Web search can stay on the current subagent path until synchronous `per_call`
  wrapping or the tool queue is proven.

## Deferred Decisions

These are intentionally deferred beyond the first implementation cycle:

- Whether role allowlists and approval requirements should move from repository
  config to an audited admin-managed policy store.
- Whether filtered per-caller MCP discovery is needed before exposing
  confidential tools.
- Which specific external MCP servers qualify as `platform_trusted`.
- Whether long-running tools should default to `queued_wait` or
  `queued_detached` after the tool queue exists.

## Acceptance Criteria For Phase 0/1

- Selected pilot tools can be loaded from platform config.
- Generic LangChain wrappers work outside Mycroft.
- At least one LangGraph agent can receive platform tools through an explicit
  additive injection contract.
- Mycroft pilot integration, if enabled, uses the same generic adapter with no
  Mycroft-specific logic inside `platform_tools`.
- Direct internal tool calls remain backward compatible.
- Agent-specific state tools are untouched.
- Tool policy blocks unauthorized roles/scopes.
- Tool audit logs contain no raw secrets or bearer tokens.
- Store artifact tool works without `ToolRuntime.state`.
- KPI, marketing, and artifact storage can run without a platform MCP service.
- Web search is either left on the current subagent path or migrated as a
  synchronous `per_call` platform tool.
- Tests cover config, wrappers, policy, context, and at least one real tool.

## Acceptance Criteria For First MCP Release

- Platform MCP service exposes selected tools through streamable HTTP.
- OAuth introspection is fail-closed.
- MCP service does not log bearer tokens or full introspection payloads.
- Static MCP discovery exposes only tools approved for broad authenticated
  discovery.
- Call-time policy blocks unauthorized roles/scopes.
- Synchronous tool calls run without blocking the MCP event loop.
- Approval-required calls fail closed when no MCP elicitation provider is
  available.
