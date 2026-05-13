# Phase 3 - Execution Safety Implementation Design

Status: design draft for implementation.

Date: 2026-05-08

Related documents:

- `docs/guardrails/comprehensive_guardrails_architecture_vision_v2.md`
- `docs/guardrails/implementation_status.md`

## 1. Purpose

Phase 1 added the reusable privacy, context, middleware, and audit primitives.
Phase 2 added scanner enforcement for the `artifact_creator_agent` sample path.

Phase 3 adds execution safety around the things the model can cause the
platform to do:

- tool authorization;
- tool argument validation and optional privacy transforms;
- tool result scanning and minimization;
- retrieval result scanning and trust labeling;
- deterministic SQL safety validation;
- sensitive row and field minimization before data is returned to models,
  files, logs, or downstream tools.

The initial production-shaped integration target remains
`artifact_creator_agent`. Other agents should be migrated only after the common
execution-safety APIs have focused regression tests.

## 2. Design Constraints From Current Architecture

### 2.1 Palimpsest remains the only reversible privacy rail

The implementation must preserve the current architectural decision:

```text
Palimpsest = sole reversible anonymizer/de-anonymizer
LLM Guard = scanner orchestration and security scanner layer
```

LLM Guard `Anonymize` and `Deanonymize` must remain forbidden in production
runtime profiles.

### 2.2 Retrieval itself is not anonymized

The current `sd_ass_agent` retrieval stack under `agents/retrievers` is
deprecated for future guarded retrieval work.

New retrieval work must target `rag_lib`, using `services.kb_manager.gaz_runtime`
as the local reference implementation. That service already builds a
manifest, stores `rag_lib.core.domain.Segment` objects in `LocalPickleStore`,
hydrates search hits through `create_scored_dual_storage_retriever`, and returns
structured search/read payloads.

Phase 3 must not add Palimpsest anonymization inside the retrieval operation:

```text
raw or scanner-sanitized query
  -> rag_lib retrieval
  -> retrieved chunks
  -> retrieval scanner/trust labeling
  -> context assembly or tool result
  -> Palimpsest anonymization at the model/tool guard boundary
```

This means:

- retrieval queries are not Palimpsest-anonymized before vector search;
- retrieved chunk text is not Palimpsest-anonymized inside the retrieval module;
- retrieved chunks that are passed into model context are anonymized later by
  `PrivacyModelRequestMiddleware`;
- retrieved chunks returned through tools can be anonymized by the tool guard
  according to that tool profile.

Deployment note: if a `rag_lib` embedding or reranking provider is external,
raw retrieval input may leave the trusted environment. That is a deployment
provider decision, not something Phase 3 should silently hide with
anonymization, because the stated product requirement is not to anonymize
retrieval itself.

### 2.3 Tool privacy transforms are profile-driven and optional

Tool guardrails must support optional anonymization/de-anonymization of tool
arguments and tool results around tool execution.

Examples:

- a CRM lookup tool may need de-anonymized arguments before execution;
- a retrieval tool may accept raw user terms and return raw internal excerpts,
  but its result should be anonymized before it re-enters model context;
- a file/artifact commit tool should receive de-anonymized text so the stored
  artifact is real, while the tool message returned to the LLM should remain
  anonymized;
- a privacy-preserving external API tool may require anonymized arguments
  before execution and no de-anonymization afterward.

## 3. Execution Split

Phase 3 should be delivered as three sequential implementation subphases. Each
subphase must be independently reviewable and testable.

### 3.1 Phase 3A - Tool Execution Safety

Scope:

- `platform_guardrails/tool_policy.py`;
- `ToolExecutionSafetyMiddleware`;
- profile-driven tool authorization;
- profile-driven tool argument/result privacy transforms;
- tool argument scanning;
- tool result scanning and minimization;
- `artifact_creator_agent` sample wiring.

Exit criteria:

- `artifact_creator_agent` no longer relies on the argument-only
  `ToolContentScannerMiddleware` for guarded tool execution;
- every guarded tool call has a profile or is blocked;
- `commit_artifact_final_text` stores real artifact text while returning safe
  model-visible tool messages;
- tool policy and artifact integration tests pass.

### 3.2 Phase 3B - Retrieval Safety

Scope:

- `platform_guardrails/retrieval.py`;
- `rag_lib`/LangChain document result normalization;
- GAZ runtime payload normalization;
- source trust labeling;
- per-chunk scanner enforcement;
- safe context rendering.

Exit criteria:

- retrieval guard APIs work with `rag_lib` document-style results;
- retrieval itself is not Palimpsest-anonymized;
- blocked chunks are excluded from model context;
- retrieval audit logs contain scanner metadata but no raw chunk text.

### 3.3 Phase 3C - SQL Safety

Scope:

- `platform_guardrails/sql_policy.py`;
- deterministic SQL parsing and validation;
- read-only enforcement;
- table/column/sensitive-field policy;
- row and field minimization;
- guarded integration into `agents/sql_query_gen.py` and BI/reporting flows
  behind explicit configuration.

Exit criteria:

- unsafe SQL is rejected before execution;
- generated and repaired SQL are both validated;
- sensitive rows/fields can be minimized before answer generation and export;
- SQL policy tests and existing BI tests pass.

## 4. Goals

1. Add common Phase 3 modules:

```text
platform_guardrails/tool_policy.py
platform_guardrails/retrieval.py
platform_guardrails/sql_policy.py
```

2. Add a tool execution middleware that can:

- resolve a tool security profile;
- enforce role and allowlist policy;
- validate arguments against the declared schema/profile;
- scan model-generated tool arguments before execution;
- apply optional argument privacy transforms before the tool runs;
- execute the tool only if policy allows;
- scan tool outputs before they re-enter graph/model context;
- minimize large or sensitive result payloads;
- apply optional result privacy transforms;
- log structured guardrail decisions without raw text or mappings.

3. Add a retrieval guard that can:

- accept `rag_lib`/LangChain document-like results;
- label trust by source type;
- scan retrieved chunk text for prompt injection, secrets, malicious URLs,
  toxicity, and banned topics according to the scanner profile;
- drop blocked chunks;
- return a structured `GuardedChunk` list for downstream context assembly;
- avoid Palimpsest anonymization inside retrieval itself.

4. Add SQL safety validation that can:

- deterministically reject non-read-only SQL;
- reject multiple statements;
- reject filesystem/network/database extension operations;
- enforce table and column policy;
- enforce row limits when configured;
- minimize result rows before answer generation or file export;
- validate generated and repaired SQL before every execution attempt.

5. Wire `artifact_creator_agent` as the sample integration:

- replace the current argument-only `ToolContentScannerMiddleware` with the new
  tool execution guard;
- keep `commit_artifact_final_text` behavior: real artifact text is stored, but
  tool messages returned to the model are privacy-protected;
- keep Phase 1 and Phase 2 scanner/privacy behavior unchanged for model calls.

## 5. Non-Goals

Phase 3 should not:

- replace Palimpsest;
- enable LLM Guard anonymization scanners;
- migrate every legacy agent;
- fully migrate `sd_ass_agent` to `rag_lib`;
- add policy-as-code YAML as the source of truth;
- add final answer groundedness checks;
- certify Russian scanner model quality.

Those are separate rollout or later-phase tasks.

## 6. Target Runtime Flow

### 6.1 Model call flow, unchanged from Phase 2

```text
messages/state/system prompt
  -> SecurityScannerMiddleware
  -> PrivacyModelRequestMiddleware
  -> model call
  -> PrivacyModelRequestMiddleware de-anonymization if allowed
  -> SecurityScannerMiddleware output scan
  -> graph state/user response
```

### 6.2 Tool execution flow, new in Phase 3

```text
model-generated tool call
  -> resolve GuardrailContext
  -> resolve ToolSecurityProfile
  -> role/allowlist/side-effect policy
  -> argument schema/profile validation
  -> scan model-generated arguments
  -> optional argument privacy transform
  -> execute tool
  -> extract textual/result payloads
  -> scan tool result as untrusted context
  -> minimize result payload
  -> optional result privacy transform
  -> return safe ToolMessage/Command/update to graph
```

### 6.3 Retrieval flow, new common guard

```text
query
  -> rag_lib retrieval runtime
  -> document-like hits
  -> normalize to RetrievedChunk
  -> source trust policy
  -> per-chunk scanner rail
  -> drop/rewrite/redact according to scanner decisions
  -> GuardedChunk list
  -> context assembly or tool result
  -> later Palimpsest model/tool-boundary transform
```

### 6.4 SQL flow, new common guard

```text
user question
  -> SQL-generation LLM
  -> validate generated SQL
  -> execute only if allowed
  -> if DB error, repair SQL
  -> validate repaired SQL
  -> execute only if allowed
  -> minimize rows and sensitive columns
  -> answer-generation LLM
  -> optional file/image export governed by context policy
```

## 7. New Module: `platform_guardrails/tool_policy.py`

### 7.1 Core data types

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

ToolCategory = Literal[
    "retrieval",
    "web_search",
    "database",
    "file_export",
    "notification",
    "write_action",
    "internal_state",
]

SideEffect = Literal["none", "read", "write", "external"]
DataSensitivity = Literal["public", "internal", "confidential", "regulated"]
PrivacyTransform = Literal["none", "anonymize", "deanonymize"]
UnprofiledToolPolicy = Literal["block", "allow_read_only"]


@dataclass(frozen=True)
class ToolPrivacyProfile:
    argument_transform: PrivacyTransform = "none"
    result_transform: PrivacyTransform = "none"
    transform_command_messages_only: bool = True
    preserve_command_update_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolResultPolicy:
    scan_result: bool = True
    max_text_chars: int = 12000
    max_items: int = 30
    allowed_result_keys: tuple[str, ...] = ()
    denied_result_keys: tuple[str, ...] = ()
    sensitive_result_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSecurityProfile:
    name: str
    category: ToolCategory
    side_effect: SideEffect
    data_sensitivity: DataSensitivity = "internal"
    allowed_roles: tuple[str, ...] = ("default",)
    requires_approval: bool = False
    allow_external_network: bool = False
    allow_file_export: bool = False
    privacy: ToolPrivacyProfile = field(default_factory=ToolPrivacyProfile)
    result_policy: ToolResultPolicy = field(default_factory=ToolResultPolicy)
```

### 7.2 Policy rail

The policy rail should convert every policy decision into the common
`GuardrailDecision` contract.

Required checks:

- unprofiled tools are blocked by default for guarded agents;
- `context.user_role` must be in `profile.allowed_roles`;
- `profile.allow_external_access` requires `context.allow_external_tool_access`;
- `profile.allow_file_export` requires `context.allow_file_export`;
- `profile.data_sensitivity in {"confidential", "regulated"}` requires
  `context.allow_sensitive_data`, unless the profile explicitly allows the role;
- `requires_approval=True` returns `review` until a real approval mechanism is
  available.

Implementation shape:

```python
class ToolPolicyRail:
    def __init__(
        self,
        profiles: Mapping[str, ToolSecurityProfile],
        *,
        unprofiled_tools: UnprofiledToolPolicy = "block",
    ) -> None:
        ...

    def profile_for(self, tool_name: str) -> ToolSecurityProfile | None:
        ...

    def evaluate_call(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        context: GuardrailContext,
    ) -> GuardrailDecision:
        ...
```

### 7.3 Tool execution middleware

Add a new middleware:

```python
class ToolExecutionSafetyMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        policy_rail: ToolPolicyRail,
        scanner_rail: LLMGuardScannerRail | None = None,
        privacy_rail: PrivacyRail | None = None,
        agent_name: str = "unknown",
        event_logger: GuardrailEventLogger | None = None,
        event_log_path: str | None = None,
    ) -> None:
        ...
```

This middleware should replace `ToolContentScannerMiddleware` for new guarded
integrations. `ToolContentScannerMiddleware` can remain as a backward-compatible
argument scanner until all guarded agents migrate.

The new middleware owns the complete tool lifecycle. To avoid duplicate privacy
transforms, `PrivacyModelRequestMiddleware` should get a compatibility option:

```python
PrivacyModelRequestMiddleware(..., guard_tool_calls: bool = True)
```

For Phase 3 guarded agents:

```text
PrivacyModelRequestMiddleware(guard_tool_calls=False)
ToolExecutionSafetyMiddleware(privacy_rail=privacy_rail, ...)
```

For legacy guarded agents:

```text
PrivacyModelRequestMiddleware(guard_tool_calls=True)
ToolContentScannerMiddleware(...)
```

### 7.4 Tool argument scanning

Tool arguments are model-generated output, so they should be scanned with the
scanner rail's output profile, matching current Phase 2 behavior.

Boundary:

```text
tool_arguments
```

If a scanner returns:

- `allow`: pass sanitized value onward;
- `redact`: pass sanitized value onward;
- `review` or `block`: do not execute the tool and remove the model tool-call
  message from state when possible.

### 7.5 Tool result scanning

Tool results are untrusted context entering the graph/model, even when the tool
is internal. They should be scanned before being returned to model context.

Recommended scanner use:

- prompt injection: input scanner;
- secrets: input scanner;
- malicious URLs: output scanner if the result contains generated/recommended
  URLs, with source URL memory preserved;
- toxicity/banned topics: input or output scanner according to scanner support.

The practical implementation can start with:

```python
scanner_rail.scan_input_text(result_text, context, boundary="tool_result")
```

and add a dedicated `scan_tool_result_text(...)` profile later if input/output
scanner differences become important.

### 7.6 Tool result minimization

The middleware must support result minimization before model context:

- truncate large strings by configured character budget;
- limit list length;
- drop denied keys;
- keep only allowed keys when `allowed_result_keys` is configured;
- replace sensitive keys with summaries;
- preserve non-message `Command.update` keys only when the profile allows it.

This is important for tools that return structured payloads, SQL rows, or raw
retrieved excerpts.

### 7.7 Artifact creator sample profile

Initial built-in profile:

```python
ARTIFACT_CREATOR_TOOL_PROFILES = {
    "commit_artifact_final_text": ToolSecurityProfile(
        name="commit_artifact_final_text",
        category="internal_state",
        side_effect="write",
        data_sensitivity="internal",
        allowed_roles=("default", "service_desk", "sales_manager"),
        requires_approval=False,
        privacy=ToolPrivacyProfile(
            argument_transform="deanonymize",
            result_transform="anonymize",
            transform_command_messages_only=True,
            preserve_command_update_keys=("artifacts", "phase"),
        ),
        result_policy=ToolResultPolicy(
            scan_result=True,
            max_text_chars=2000,
            allowed_result_keys=(),
            denied_result_keys=(),
        ),
    ),
}
```

Expected behavior:

- `final_text` is de-anonymized before `commit_artifact_final_text` runs;
- `Command.update["artifacts"]` stores the real artifact text;
- `Command.update["messages"]` is anonymized before returning to model context;
- malicious or banned content in `final_text` can still be blocked before the
  tool runs by argument scanning;
- unsafe tool-result text is blocked or sanitized before it re-enters the graph.

### 7.8 Optional extra tools in `artifact_creator_agent`

`initialize_agent(..., tools=[...])` currently accepts arbitrary extra tools.
For guarded execution, extra tools need profiles.

Add optional parameters:

```python
guardrail_tool_profiles: Mapping[str, ToolSecurityProfile | Mapping[str, Any]] | None = None
guardrail_unprofiled_tools: Literal["block", "allow_read_only"] = "block"
```

When `guardrails_enabled=True`, unprofiled extra tools should be blocked by
default. This is safer and makes tool exposure explicit.

## 8. New Module: `platform_guardrails/retrieval.py`

### 8.1 Core data types

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

SourceType = Literal["kb", "ticket", "web", "database", "tool", "unknown"]
TrustLevel = Literal["trusted", "internal", "external", "untrusted"]


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    source_type: SourceType = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardedChunk:
    text: str
    source: str
    source_type: SourceType
    trust_level: TrustLevel
    contains_pii: bool
    injection_score: float
    allowed: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    decisions: tuple[GuardrailDecision, ...] = ()
```

### 8.2 Source trust policy

Default mapping:

| Source type | Trust level | Notes |
| --- | --- | --- |
| `kb` | `trusted` | Internal curated KB, still scanned. |
| `ticket` | `internal` | Internal operational content, may contain PII. |
| `database` | `internal` | Structured internal data, row policy still applies. |
| `web` | `external` | Treat as untrusted instructions. |
| `tool` | `internal` | Depends on tool profile. |
| `unknown` | `untrusted` | Safe default. |

### 8.3 rag_lib normalization

The guard must accept either LangChain `Document` objects returned from
`rag_lib` retrievers or structured payloads such as those returned by
`GazRuntimeService.search_sales_materials(...)` and
`GazRuntimeService.read_material(...)`.

Required normalizers:

```python
def chunk_from_document(document: Any) -> RetrievedChunk:
    text = getattr(document, "page_content", "") or ""
    metadata = dict(getattr(document, "metadata", {}) or {})
    source = (
        metadata.get("relative_path")
        or metadata.get("source")
        or metadata.get("segment_id")
        or ""
    )
    source_type = metadata.get("source_type") or infer_source_type(metadata)
    return RetrievedChunk(...)


def chunks_from_gaz_payload(payload: Mapping[str, Any]) -> list[RetrievedChunk]:
    ...
```

For `read_material`, each item under `excerpts[]` maps naturally to one
`RetrievedChunk`.

For `search_sales_materials`, candidate previews can be guarded as short chunks,
but deep content scanning should happen on `read_material` excerpts, where the
actual text that will reach the model is available.

### 8.4 Retrieval guard rail

```python
class RetrievalGuardrail:
    def __init__(
        self,
        scanner_rail: LLMGuardScannerRail,
        *,
        event_logger: GuardrailEventLogger | None = None,
        event_log_path: str | None = None,
        fail_closed: bool = True,
    ) -> None:
        ...

    def guard_chunks(
        self,
        chunks: Sequence[RetrievedChunk],
        context: GuardrailContext,
        *,
        boundary: str = "retrieval_chunk",
    ) -> list[GuardedChunk]:
        ...
```

Rules:

- scan each chunk text before it is assembled into context;
- if scanner blocks a chunk, set `allowed=False` and do not include it in
  assembled model context;
- if scanner redacts a chunk, include sanitized text and preserve decision
  metadata;
- always label source type and trust level;
- never call `PrivacyRail.anonymize_text(...)` here;
- log scanner decisions without raw chunk text.

### 8.5 Retrieval context assembly

Provide a helper that renders only allowed chunks:

```python
def render_guarded_context(chunks: Sequence[GuardedChunk]) -> str:
    ...
```

Suggested rendering:

```text
[source_type=kb trust=trusted source=...]
chunk text
```

This labels retrieved content as data, not instructions. It does not replace
prompt-level instruction hierarchy, but it helps downstream prompts keep source
boundaries clear.

## 9. New Module: `platform_guardrails/sql_policy.py`

### 9.1 Dependency

Use `sqlglot` for deterministic parsing and AST inspection.

If `sqlglot` is not available, guarded SQL execution should fail closed when
SQL guardrails are enabled.

### 9.2 Core data types

```python
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SQLPolicy:
    dialect: str = "sqlite"
    allowed_tables: tuple[str, ...] = ()
    denied_tables: tuple[str, ...] = ()
    denied_columns: tuple[str, ...] = ()
    sensitive_columns: tuple[str, ...] = ()
    denied_functions: tuple[str, ...] = (
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "sqlite_attach",
        "load_extension",
    )
    require_limit: bool = True
    default_limit: int = 100
    max_limit: int = 1000
    allow_sensitive_columns_roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class SQLValidationResult:
    query: str
    decision: GuardrailDecision
    normalized_query: str | None = None
```

### 9.3 Validation rules

The validator must reject:

- multiple statements;
- non-`SELECT` statements;
- DDL/DML commands such as `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`,
  `CREATE`, `TRUNCATE`, `MERGE`;
- `COPY`, `ATTACH`, `DETACH`, `LOAD`, `INSTALL`, `PRAGMA` when unsafe;
- filesystem or network functions;
- blocked tables;
- sensitive columns unless role/context permits;
- `SELECT *` when sensitive/denied columns are configured and cannot be proven
  safe;
- queries without `LIMIT` when `require_limit=True`, unless the query is a pure
  aggregate returning bounded output.

The first implementation may fail closed on missing limits rather than rewrite
SQL. A second step can add safe limit injection using `sqlglot` AST rewriting.

### 9.4 Integration with `agents/sql_query_gen.py`

Current flow:

```python
query = write_query(question, db, ...)
for attempt in range(3):
    result = execute_query(query, db)
    if result.get("error") is None:
        break
    query = fix_query(query, result["error"])
```

Guarded flow:

```python
query = write_query(question, db, ...)
for attempt in range(3):
    validation = sql_guard.validate(query, context)
    if not validation.decision["allowed"]:
        raise SQLPolicyViolation(query, validation.decision)
    query_to_execute = validation.normalized_query or validation.query

    result = execute_query(query_to_execute, db)
    if result.get("error") is None:
        break
    query = fix_query(query_to_execute, result["error"])
```

The repaired query must be validated again before execution.

### 9.5 Row and field minimization

Add a row minimizer:

```python
def minimize_rows(
    rows: Sequence[Mapping[str, Any]],
    policy: SQLPolicy,
    context: GuardrailContext,
    *,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    ...
```

Rules:

- drop denied columns;
- drop or summarize sensitive columns unless allowed by role/context;
- cap number of rows visible to answer generation;
- cap number of rows included in exported files unless file export policy
  explicitly permits full export;
- log only counts, column names after minimization, and decision metadata.

`answer_row_limit` in the existing BI path remains useful, but it is not a
security control by itself. SQL policy should enforce security-sensitive caps
before answer generation and export.

## 10. Middleware Ordering For `artifact_creator_agent`

Current guarded drafting middleware:

```text
build_prompt
SecurityScannerMiddleware
PrivacyModelRequestMiddleware
ToolContentScannerMiddleware
```

Target Phase 3 drafting middleware:

```text
build_prompt
SecurityScannerMiddleware
PrivacyModelRequestMiddleware(guard_tool_calls=False)
ToolExecutionSafetyMiddleware
```

Model-call behavior stays with the first two guardrail middlewares. Tool-call
behavior moves to `ToolExecutionSafetyMiddleware`.

Confirmation agent target:

```text
build_prompt
SecurityScannerMiddleware
provider_then_tool
PrivacyModelRequestMiddleware(guard_tool_calls=False)
ToolExecutionSafetyMiddleware only if confirmation tools are added
```

The current confirmation agent has no external tools beyond structured output,
so Phase 3 does not need a tool guard there unless future tools are added.

## 11. Backward Compatibility Plan

1. Keep `ToolContentScannerMiddleware` exported for existing tests and any
   agent still using it.
2. Add `ToolExecutionSafetyMiddleware` and new module exports.
3. Add `guard_tool_calls=True` default to `PrivacyModelRequestMiddleware`.
4. Switch only `artifact_creator_agent` to `guard_tool_calls=False` plus
   `ToolExecutionSafetyMiddleware`.
5. Do not change `sd_ass_agent` legacy retrieval in Phase 3. Design new
   retrieval guard APIs around `rag_lib`.

## 12. Configuration Additions

`artifact_creator_agent.initialize_agent(...)` should add:

```python
guardrail_tool_profiles: Mapping[str, Any] | None = None
guardrail_unprofiled_tools: Literal["block", "allow_read_only"] = "block"
guardrail_sql_enabled: bool = False
guardrail_retrieval_enabled: bool = False
```

Only `guardrail_tool_profiles` and `guardrail_unprofiled_tools` are expected to
be used by the artifact creator sample immediately.

For BI/reporting agents, guarded SQL should be configured through `init_context`
or future policy files:

```json
{
  "guardrail_sql_enabled": true,
  "sql_policy": {
    "dialect": "sqlite",
    "require_limit": true,
    "default_limit": 100,
    "max_limit": 1000,
    "sensitive_columns": ["phone", "email", "passport", "snils"]
  }
}
```

Policy-as-code YAML remains Phase 5. Phase 3 can use Python dataclasses and
JSON-compatible config parsing.

## 13. Test Plan

### 13.1 Tool policy tests

Add `tests/unit/test_platform_guardrails_tool_policy.py`.

Required cases:

- role-allowed tool executes;
- role-denied tool returns block and handler is not called;
- unprofiled tool blocks by default;
- `requires_approval=True` returns review and handler is not called;
- external tool requires `allow_external_tool_access=True`;
- file export tool requires `allow_file_export=True`;
- confidential/regulated tool requires sensitive-data permission or allowed
  role;
- blocked tool-call AI message is removed from graph state when possible;
- audit log does not contain raw arguments.

### 13.2 Tool privacy tests

Required cases:

- `argument_transform="deanonymize"` de-anonymizes args before handler;
- `argument_transform="anonymize"` anonymizes args before handler;
- `result_transform="anonymize"` anonymizes tool message content;
- `transform_command_messages_only=True` preserves non-message command updates;
- artifact commit stores raw `artifact_final_text` but returns anonymized
  `ToolMessage`;
- privacy transforms are not duplicated when
  `PrivacyModelRequestMiddleware(guard_tool_calls=False)` is used.

### 13.3 Tool result scanner tests

Required cases:

- prompt injection in tool result blocks result before model context;
- secrets in tool result are redacted and continue;
- generated malicious URL in tool result blocks;
- source-provided URL memory still allows preserving sourced URLs;
- large result payload is minimized.

### 13.4 Retrieval tests

Add `tests/unit/test_platform_guardrails_retrieval.py`.

Required cases:

- LangChain/rag_lib document normalizes to `RetrievedChunk`;
- GAZ `read_material` payload normalizes to chunks;
- source type maps to expected trust level;
- safe chunks are allowed;
- blocked chunks are excluded from rendered context;
- redacted scanner output updates chunk text;
- retrieval guard never calls `PrivacyRail`;
- guardrail audit logs contain scanner metadata but not raw chunk text.

### 13.5 SQL policy tests

Add `tests/unit/test_platform_guardrails_sql_policy.py`.

Required cases:

- `SELECT ... LIMIT` is allowed;
- `WITH ... SELECT` is allowed;
- `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER` are blocked;
- multiple statements are blocked;
- `ATTACH`, `COPY`, unsafe `PRAGMA`, and extension load functions are blocked;
- blocked table is rejected;
- denied column is rejected;
- sensitive column is rejected for default role and allowed for configured role;
- missing limit is blocked or safely rewritten;
- repaired SQL is revalidated before execution;
- row minimizer drops sensitive fields and caps rows.

### 13.6 Artifact creator integration tests

Extend `tests/unit/test_artifact_creator_agent.py`.

Required cases:

- initialization wires `ToolExecutionSafetyMiddleware` when guardrails are
  enabled;
- `commit_artifact_final_text` has the built-in profile;
- final artifact storage still receives raw text;
- tool message returned to model is anonymized;
- unauthorized extra tool is blocked when no profile is supplied;
- profiled extra read-only tool can execute for allowed role.

## 14. Implementation Steps

### 14.1 Phase 3A - Tools

1. Implement `tool_policy.py` dataclasses, registry, evaluation logic, and
   result minimization helpers.
2. Implement `ToolExecutionSafetyMiddleware` in `middleware.py` or a dedicated
   `tool_middleware.py`, then export it from `platform_guardrails.__init__`.
3. Add `guard_tool_calls` option to `PrivacyModelRequestMiddleware`.
4. Wire `artifact_creator_agent` to the new middleware and built-in
   `commit_artifact_final_text` profile.
5. Add focused tool policy, tool privacy, tool result scanner, and artifact
   integration tests.
6. Run the Phase 3A focused test set:

```powershell
uv run pytest -q `
  tests\unit\test_platform_guardrails_middleware.py `
  tests\unit\test_platform_guardrails_scanners.py `
  tests\unit\test_platform_guardrails_tool_policy.py `
  tests\unit\test_artifact_creator_agent.py
```

### 14.2 Phase 3B - Retrieval

1. Implement `retrieval.py` normalizers and `RetrievalGuardrail`.
2. Add `rag_lib`/LangChain document normalization tests.
3. Add GAZ runtime payload normalization tests.
4. Add scanner/trust-labeling tests using fake `Document` and GAZ payloads.
5. Keep retrieval-level Palimpsest anonymization out of this subphase.
6. Run the Phase 3B focused test set:

```powershell
uv run pytest -q `
  tests\unit\test_platform_guardrails_scanners.py `
  tests\unit\test_platform_guardrails_retrieval.py `
  tests\unit\test_gaz_runtime.py
```

### 14.3 Phase 3C - SQL

1. Implement `sql_policy.py` with `sqlglot` validation and row minimization.
2. Add SQL policy tests.
3. Integrate SQL validation into `agents/sql_query_gen.py` behind an explicit
   guarded parameter.
4. Validate generated and repaired SQL before every execution attempt.
5. Update BI agent guarded configuration only after the common SQL policy tests
   pass.
6. Run the Phase 3C focused test set:

```powershell
uv run pytest -q `
  tests\unit\test_platform_guardrails_sql_policy.py `
  tests\unit\test_sql_query_gen.py
```

### 14.4 Full Phase 3 Regression

After all three subphases pass independently, run the combined guardrail and
affected-agent regression set:

```powershell
uv run pytest -q `
  tests\unit\test_platform_guardrails_middleware.py `
  tests\unit\test_platform_guardrails_scanners.py `
  tests\unit\test_platform_guardrails_tool_policy.py `
  tests\unit\test_platform_guardrails_retrieval.py `
  tests\unit\test_platform_guardrails_sql_policy.py `
  tests\unit\test_artifact_creator_agent.py `
  tests\unit\test_sql_query_gen.py
```

## 15. Acceptance Criteria

### 15.1 Phase 3A - Tools

Tools are complete when:

- `artifact_creator_agent` uses a common tool execution guard;
- every guarded tool call has a resolved profile or is blocked;
- role/tool policy decisions are logged;
- tool arguments are scanner-checked before execution;
- tool arguments and results support profile-driven Palimpsest transforms;
- tool outputs are scanned before re-entering model context;
- artifact storage still preserves the real final artifact text;
- tool guard logs avoid raw tool payloads and Palimpsest mappings.

### 15.2 Phase 3B - Retrieval

Retrieval is complete when:

- retrieval guard APIs work with `rag_lib` document-style results and do not
  anonymize retrieval itself;
- GAZ runtime payloads can be normalized into guarded chunks;
- retrieved chunks are trust-labeled by source type;
- blocked chunks are excluded from rendered model context;
- retrieval logs avoid raw retrieved chunks and Palimpsest mappings.

### 15.3 Phase 3C - SQL

SQL is complete when:

- SQL guard APIs reject unsafe SQL deterministically;
- generated and repaired SQL are validated before execution;
- SQL result minimization can remove sensitive fields and cap rows;
- SQL logs avoid raw result rows and Palimpsest mappings.

### 15.4 Full Phase 3

Phase 3 is complete when all three subphase exit criteria are met and the full
Phase 3 regression set passes.

## 16. Open Decisions

1. Approval UX: until there is a human approval workflow, should
   `requires_approval=True` map to `review` or hard `block`? Recommended Phase 3
   default: `review`.
2. SQL limit policy: should missing `LIMIT` be blocked or rewritten? Recommended
   first implementation: block, then add AST rewrite after tests.
3. Sensitive SQL columns: should defaults be regex-based or explicit per agent?
   Recommended Phase 3 default: explicit policy list, with regex helpers only as
   optional config.
4. Retrieval PII flag: should `contains_pii` be scanner-based or Palimpsest dry
   detection? Recommended Phase 3 default: `False` unless a non-reversible
   detector is configured, because retrieval must not mutate text through
   Palimpsest.
5. Unprofiled tools: should guarded agents ever allow read-only unprofiled
   tools? Recommended default: block. Allow read-only only in local development
   or explicit test profiles.
