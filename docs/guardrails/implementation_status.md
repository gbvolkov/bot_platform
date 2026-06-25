# Guardrails Implementation Status

Current status as of 2026-06-09.

This document describes what is implemented in the repository now. It is
separate from the target architecture vision, which intentionally includes later
phases that are not yet implemented.

## Scope

The implemented guardrail path is the platform reusable guardrail layer plus the
sample integration in `artifact_creator_agent`.

Phase 1 foundation primitives are implemented, but not every legacy agent has
been migrated to the new middleware. Phase 2 scanner enforcement, deterministic
URL/domain policy, and Phase 3A tool execution safety are implemented for
`artifact_creator_agent` and covered by focused regression tests.

Phase 2 should not yet be treated as fully production-complete for all
production workloads. The platform wiring is in place. Russian and mixed
Russian/English prompt-injection benchmarking and representative evaluation set
creation are complete for the current workstream, and the configured
prompt-injection thresholds are configured as `0.65` for model input and
`0.82` for tool-result scanning. Remaining
production hardening is now focused on rollout monitoring, regression additions
for newly discovered misses, and any future fine-tuning or model replacement if
new evaluation data shows recall is insufficient.

## Phase 1 - Foundation

Status: implemented as reusable platform primitives; partially adopted by
agents.

### Implemented Modules

`platform_guardrails/context.py`

- Builds `GuardrailContext` from LangGraph runtime/config/state.
- Tracks `tenant_id`, `user_id`, `user_role`, `thread_id`, `agent_name`,
  `request_id`, risk level, and allow flags.
- Provides `privacy_scope_key(...)` for stable Palimpsest session scoping.

`platform_guardrails/decisions.py`

- Defines `GuardrailDecision`.
- Supports `allow`, `block`, `redact`, `rewrite`, `review`, and `fallback`
  actions.

`platform_guardrails/privacy.py`

- Wraps Palimpsest in `PrivacyRail`.
- Keeps Palimpsest as the sole reversible anonymizer/de-anonymizer.
- Scopes privacy sessions by `tenant_id | user_id | thread_id`.
- Provides reset handling for thread reset messages.
- Performs a runtime preflight for required Palimpsest language dependencies.

`platform_guardrails/middleware.py`

- Provides `PrivacyModelRequestMiddleware`.
- Anonymizes model request messages and dynamic system prompts before external
  model calls.
- De-anonymizes AI model responses after the model call when
  `allow_deanonymization=True`.
- De-anonymizes tool arguments before execution and anonymizes tool message
  results before they re-enter model context.
- Preserves non-message command state such as artifact payloads, so artifact
  storage can keep the real generated artifact.
- Retains `PalimpsestSessionMiddleware` as a backward-compatible adapter for
  legacy agents.

`platform_guardrails/logging.py`

- Provides `RedactingJSONFileTracer`.
- Logs prompt/output lengths, not raw prompt/output text.
- Provides `GuardrailEventLogger` for structured guardrail decision audit logs.

### Implemented Observability Policy

- Local JSON callback tracing is redacted by default.
- Guardrail decision logs include scanner names, actions, scores, categories,
  and boundary metadata, but not raw prompt text or raw secrets.
- Langfuse is treated as trusted platform observability when configured; this
  is a deployment trust assumption, not a privacy transformation guarantee.

### Phase 1 Adoption

`artifact_creator_agent` uses the Phase 1 privacy middleware when
`guardrail_privacy_enabled=True`.

Other agents may still use older patterns or no guardrail middleware. Migrating
all agents to the common foundation remains rollout work, not a Phase 2 scanner
enforcement task.

### Transforming Any Agent For Phase 1 Guardrails

The Phase 1 transformation makes an agent privacy-aware without changing its
domain behavior. The required outcome is that user text, dynamic system prompts,
and model-visible tool context are anonymized before they leave the platform,
and model responses are de-anonymized before returning to the user when the
agent is allowed to restore private values.

For a platform-compiled migration, the agent module should expose
`build_agent_graph(...)` returning an `AgentGraphSpec` built with
`PlatformStateGraph`. Model nodes should be declared with `add_agent_node(...)`
instead of constructing a LangChain agent inline, and ordinary graph nodes that
handle user-controlled state should be declared with guardrails enabled or with
an explicit `NodeGuardrailPolicy`. The registry entry then selects:

```json
"guardrails": {
  "policy": "default_guardrails",
  "mode": "platform"
}
```

At startup, `bot_service.agent_registry` builds a
`PlatformGuardrailRuntime.from_policy_id(...)` and `PlatformGraphCompiler`
injects `PrivacyModelRequestMiddleware` into guarded model nodes. This is the
preferred shape for new migrations because the agent does not need to manually
construct Palimpsest sessions, event log paths, or model middleware ordering.

For an agent-owned migration, keep the existing `initialize_agent(...)` contract
but add the Phase 1 guardrail parameters:

- `guardrails_locale`
- `guardrail_privacy_enabled`
- `guardrail_palimpsest_run_entities`
- `guardrail_palimpsest_entity_replacements`
- `guardrail_palimpsest_options`
- `guardrail_palimpsest_session_options`

The agent should create one `PrivacyRail` for the run and reuse it across model
middleware and tool-result transforms. Dynamic prompts should be placed before
`PrivacyModelRequestMiddleware` in the LangChain middleware list, so the prompt
text is also anonymized before the model call. Any non-agent graph node that
persists user-controlled prompt or context state should be wrapped with
`guarded_node(..., privacy_middleware=...)`.

The registry supports the agent-owned form through `params.guardrail_policy`.
When present, `bot_service.agent_registry` resolves the policy from
`data/config/guardrails/policies.json` into `guardrail_*` initialization kwargs.
An agent must not mix `guardrail_policy` with inline `guardrail_*` params in the
load config.

Phase 1 migration is complete for an agent when:

- all external model calls pass through `PrivacyModelRequestMiddleware`;
- Palimpsest replacement configuration comes from guardrail policy rather than
  hard-coded agent constants;
- raw prompt/output text is absent from local guardrail traces;
- prompt-setting or other user-controlled state-mutating nodes are protected
  before they store data in graph state;
- regression tests confirm private values are anonymized before the model call
  and restored only at approved output boundaries.

## Phase 2 - Scanner Enforcement

Status: mostly implemented for `artifact_creator_agent`; Russian prompt-injection
scanner model selection and calibrated threshold are now configurable through
guardrail policy.

### Implemented Modules

`platform_guardrails/scanners.py`

- Defines `ScannerSpec`, `LLMGuardScannerProfile`, and `LLMGuardScannerRail`.
- Uses LLM Guard only for scanner orchestration and safety/security scanning.
- Rejects LLM Guard `Anonymize` and `Deanonymize` in configured profiles.
- Supports `fail_closed` and `fail_open` scanner failure policies.
- Defaults guarded agents to `fail_closed`.
- Supports configurable LLM Guard `PromptInjection` model config, optional
  explicit revision, and threshold. `platform_guardrails.scanners` does not
  bundle concrete model paths, thresholds, revisions, or model-specific
  compatibility defaults; those belong in deployment/agent configuration.
- Remembers URLs accepted from source input by guardrail context scope so later
  model/tool output can preserve source URLs without treating them as invented
  generated links.

`platform_guardrails/injection.py`

- Maps scanner names to categories.
- Provides Russian block/review response text.
- Converts scanner results into `GuardrailDecision` actions.

`platform_guardrails/middleware.py`

- Provides `SecurityScannerMiddleware`.
- Provides `ToolContentScannerMiddleware`.
- Provides `guarded_node(...)`, a reusable LangGraph node wrapper that applies
  scanner enforcement and Palimpsest privacy to non-agent graph nodes before the
  wrapped node mutates state.
- Removes blocked model-input messages from state with `RemoveMessage`.
- Uses `REMOVE_ALL_MESSAGES` replacement for id-less blocked messages.
- Removes unsafe model-generated tool-call messages from state when tool
  argument scanning blocks.
- For model-visible tool results, can redact LLM Guard `PromptInjection`
  sentence-level hits from untrusted tool result text and continue with the
  sanitized result after re-scanning.
- Supports composite input scanning across runtime system prompt text, recent
  human messages, and untrusted tool results. `state.system_prompt` and prior
  assistant history are not added separately to the composite text.

### Default Artifact Creator Scanner Profile

Input scanners:

- `TokenLimit`: block oversized input instead of silently truncating.
- `Secrets`: redact detected secrets and continue.
- `PromptInjection`: block when LLM Guard marks the prompt invalid; default
  matching is sentence-level to catch mixed benign/hostile prompts.
- `PromptInjection` can use a deployment-supplied Hugging Face model mapping,
  optional explicit revision, and deployment threshold. When no scanner
  threshold is supplied, LLM Guard's scanner default is used.
- Current local policy calibration for the Russian prompt-injection model uses
  `prompt_injection_threshold: 0.65` and
  `tool_result_prompt_injection_threshold: 0.82` in both `default_guardrails`
  and `persons_guardrails`.
- `Toxicity`: review/block when invalid.
- `BanTopics`: review/block for configured generic topics.

Composite input scanners:

- Default: `PromptInjection`.
- Configurable per agent through `guardrail_composite_input_scanners`.
- Runs as decision-only scanning over privileged prompt state plus a bounded
  recent message window; sanitized composite text is not written back into
  individual messages.

Output and tool-argument scanners:

- `platform_guardrails.url_policy`: config-only deterministic URL/domain policy
  that runs before `MaliciousURLs` when configured. It supports audit/enforce
  modes, exact and wildcard domain rules, private/local host blocking,
  userinfo URL blocking, mixed-script IDN checks, and protected-domain
  lookalike checks. URL matching canonicalizes IDNA/punycode hosts and Unicode
  DNS dot separators such as `。`, `．`, and `｡`, so non-Latin configured
  domains and non-Latin URLs are handled through the same deterministic path.
  Source URLs are still allowed to be preserved, but explicit deterministic
  violations override source preservation.
- `MaliciousURLs`: block when LLM Guard classifies a generated URL as malicious
  above threshold and the URL was not already present in accepted source input.
- `Toxicity`: review/block when invalid.
- `BanTopics`: review/block when invalid.

Not enabled in the default user-facing output profile:

- `Sensitive`: not used on final user-facing responses because it masks
  Palimpsest-restored client data such as names, phone numbers, emails, IDs, and
  addresses.

Explicitly forbidden while Palimpsest is active:

- `Anonymize`
- `Deanonymize`

### Artifact Creator Middleware Order

Drafting agent:

```text
build_prompt
SecurityScannerMiddleware
PrivacyModelRequestMiddleware
ToolExecutionSafetyMiddleware
```

Confirmation agent:

```text
build_prompt
SecurityScannerMiddleware
provider_then_tool
PrivacyModelRequestMiddleware
```

This gives the intended flow:

```text
raw user/tool context
  -> scanner enforcement
  -> Palimpsest anonymization
  -> model call
  -> Palimpsest de-anonymization
  -> output scanner enforcement
  -> user response or guarded tool execution
```

### Current Scanner-Policy Behavior

The current policy trusts scanner decisions. If a scanner marks content as safe,
the platform allows it, except where the configured deterministic URL/domain
policy produces an enforce-mode block. The platform does not add hidden
deterministic prompt-injection heuristics on top of LLM Guard allow decisions.

Examples:

| Input/output condition | Current behavior |
| --- | --- |
| PromptInjection marks input invalid | Block; model handler is not called; blocked message is removed from state |
| PromptInjection marks a sentence in a model-visible tool result invalid | Replace the flagged sentence with `[guarded sentence removed]`, re-scan the sanitized result, then continue if clean |
| PromptInjection scores an indirect-injection-looking pasted text as safe | Allow |
| Secrets scanner redacts a secret | Continue with sanitized text |
| TokenLimit marks input invalid | Block |
| URL policy finds a violation in `audit` mode | Allow and audit the URL-policy decision before LLM Guard URL scanning |
| URL policy finds a violation in `enforce` mode | Block before LLM Guard URL scanning |
| MaliciousURLs scores a generated URL above threshold | Block response/tool call |
| MaliciousURLs scores a password-reset lookalike URL below threshold and URL policy has no matching rule | Allow |
| URL was supplied by the user as source material and later appears in CRM/artifact text | Allow preserving it as source data unless URL policy blocks it |
| Palimpsest restores client PII in the final user-facing response | Show restored values; do not mask with LLM Guard Sensitive |
| Scanner raises and policy is `fail_closed` | Block |
| Scanner raises and policy is `fail_open` | Allow and audit scanner error |

### Transforming Any Agent For Phase 2 Guardrails

The Phase 2 transformation adds scanner enforcement around the same model and
state boundaries protected in Phase 1. The required outcome is that unsafe input
is blocked or sanitized before model execution, unsafe model output is blocked
before user delivery or tool execution, and scanner decisions are logged without
raw prompt text.

For a platform-compiled migration, guarded `PlatformStateGraph` nodes receive
`SecurityScannerMiddleware` automatically when the selected policy has
`guardrail_scanners_enabled=True`. Agent nodes should keep dynamic prompt
construction in the `prompt` argument to `add_agent_node(...)`; the compiler
orders prompt construction before scanner and privacy middleware. Callable nodes
that accept or persist user-controlled text should use the default guardrails or
an explicit `NodeGuardrailPolicy`, for example to include human and tool
messages in composite prompt-injection scans:

```python
NodeGuardrailPolicy(
    composite_message_roles=("human", "tool"),
)
```

For an agent-owned migration, add scanner parameters to `initialize_agent(...)`
and pass them into `LLMGuardScannerProfile` / `LLMGuardScannerRail`:

- `guardrail_scanners_enabled`
- `guardrail_scanner_failure_policy`
- `guardrail_banned_topics`
- `guardrail_prompt_injection_model`
- `guardrail_prompt_injection_model_revision`
- `guardrail_prompt_injection_threshold`
- `guardrail_tool_result_prompt_injection_threshold`
- `guardrail_url_policy`
- `guardrail_scan_system_prompt`
- `guardrail_verbose_logging`
- `guardrail_composite_input_scanners`
- `guardrail_composite_recent_message_limit`

The model middleware order should be:

```text
dynamic prompt builder
SecurityScannerMiddleware
PrivacyModelRequestMiddleware
tool-execution middleware, when enabled
```

Non-model nodes that store a user-supplied system prompt, routing instruction,
retrieval query, or other privileged state should be wrapped with
`guarded_node(...)` and configured to scan relevant state keys before mutation.
Blocked id-less message sets must be removed from state with
`REMOVE_ALL_MESSAGES` replacement semantics, matching the existing middleware
behavior.

URL policy should come from `data/config/guardrails/policies.json`, not from
agent code. In audit mode, deterministic URL decisions should be logged before
LLM Guard URL scanning but should not change user-visible behavior. In enforce
mode, deterministic violations block before `MaliciousURLs` runs.

Phase 2 migration is complete for an agent when:

- model input, model output, model-generated tool calls, and model-visible
  untrusted tool results are scanned at the correct boundaries;
- scanner failure policy is explicit and defaults to fail-closed for guarded
  production agents;
- dynamic system prompt text is included or excluded according to
  `guardrail_scan_system_prompt`;
- source URLs are remembered only by guardrail context scope and generated URL
  violations still block when policy requires it;
- tests cover prompt-injection block, secrets redaction, token-limit block,
  URL-policy audit/enforce behavior, scanner failure behavior, and redacted
  event logging.

## Phase 3A - Tool Execution Safety

Status: implemented for `artifact_creator_agent`.

`platform_guardrails/tool_policy.py`

- Defines `ToolSecurityProfile`, `ToolPrivacyProfile`, `ToolResultPolicy`, and
  `ToolPolicyRail`.
- Blocks unprofiled tools by default for guarded execution.
- Enforces role checks, approval-required decisions, file-export permission,
  sensitive-data permission, and the external-tool runtime switch.
- External access is not represented by a separate tool-level boolean. A tool is
  external when `category == "external_access"`. Runtime
  `allow_external_tool_access=False` blocks tools in that category.

`platform_guardrails/tool_registry.py`

- Registers concrete runtime tools against concrete runtime-name profiles.
- Builds guarded tool bundles for LangChain agent tool execution.
- Keeps one shared `PrivacyRail` available to model-request privacy middleware
  and tool-result privacy transforms, so model input anonymization and tool
  result anonymization use the same Palimpsest session scope.

`platform_tools/registry.py`

- Resolves platform tool definitions from `platform_tools/tools.json`.
- Returns a `BuiltAgentTools` bundle containing both tool objects and resolved
  runtime-name guardrail profiles.
- Validates tool profiles when tool execution guardrails are enabled.
- Fails closed when a selected MCP tool cannot be loaded. MCP connection errors
  are wrapped with the configured server name and target URL for actionable
  startup logs.

### Artifact Creator Integration

`initialize_agent(...)` accepts these guardrail parameters:

- `guardrails_locale`
- `guardrail_privacy_enabled`
- `guardrail_scanners_enabled`
- `guardrail_tool_execution_enabled`
- `guardrail_scanner_failure_policy`
- `guardrail_banned_topics`
- `guardrail_prompt_injection_model`
- `guardrail_prompt_injection_model_revision`
- `guardrail_prompt_injection_threshold`
- `guardrail_composite_input_scanners`
- `guardrail_composite_recent_message_limit`
- `guardrail_url_policy`
- `guardrail_palimpsest_run_entities`
- `guardrail_palimpsest_entity_replacements`
- `guardrail_palimpsest_options`
- `guardrail_palimpsest_session_options`
- `guardrail_tool_profiles`
- `guardrail_unprofiled_tools`

Palimpsest privacy configuration is now part of the per-agent registry contract.
Agents provide `guardrail_palimpsest_entity_replacements` as the single
session-scoped replacement matrix accepted by Palimpsest 0.1.36: each entity maps
to `fake` or `typed_placeholder`. These configured options are required: if the
installed Palimpsest API cannot accept them, initialization fails instead of
falling back to another anonymization mode.

If `guardrail_palimpsest_run_entities` is omitted, the platform derives
Palimpsest `run_entities` from the keys in
`guardrail_palimpsest_entity_replacements`. To avoid anonymizing an entity type,
exclude it from `run_entities`; with the current policy shape, that usually
means removing it from `entity_replacements`. Palimpsest 0.1.36 does not expose
a value-level allowlist through `Palimpsest(..., run_entities=...)` or
`create_session(..., entity_replacements=...)`; public-name exclusions would
require a separate platform exclusion layer or upstream Palimpsest support.

`set_prompt` is guarded with the common node wrapper when either scanner or
privacy guardrails are enabled, so user-provided system prompt text is scanned
or anonymized before it can be stored in `state.system_prompt`.

`data/config/bot_service/load.json` now references guardrails by policy id only:
`guardrail_policy: "default_guardrails"`. The policy definition lives in
`data/config/guardrails/policies.json` and independently controls privacy,
scanner, and tool-execution layers. Inline agent params such as
`guardrail_privacy_enabled` or `guardrail_tool_profiles` are intentionally not
part of the load config contract.

The local `default_guardrails` policy now resolves deterministic URL policy
configuration into `guardrail_url_policy`. The current rollout posture is audit
first: URL policy can log deterministic findings before changing user-visible
behavior, and enforce mode is available for policy-controlled blocking.

Runtime reputation checks, operational update process for URL/domain rules,
promotion from URL-policy `audit` mode to `enforce` mode, and URL-policy
monitoring are intentionally postponed. When monitoring is implemented later, it
should aggregate structured decision metadata by `rule`, `mode`, `boundary`,
action outcome, agent, and `source_url` without logging raw prompts, full
outputs, secrets, or Palimpsest mappings.

Tool execution profiles live with the tool registry in
`platform_tools/tools.json`. Internal tool templates use `guardrail_profile`;
MCP server templates use `guardrail_profiles` keyed by runtime tool name, with
unprefixed MCP names also accepted. When `tool_execution_enabled=false`, these
tool profile references are inert. When it is true, every selected runtime tool
must resolve a valid profile or startup fails.

External-tool disabling uses profile category only: profiles with
`category: "external_access"` require runtime `allow_external_tool_access=True`.
There is no separate `allow_external_access` field on tool profiles.

The Google Maps MCP template is configured with explicit streamable-HTTP
timeouts. If the external MCP endpoint is unavailable, startup fails closed with
a concise `ToolRegistryError`; configured MCP tools are not silently skipped.

Tool result anonymization is controlled independently per tool profile. Set
`anonymize_result: true` on a tool registry profile, or use
`privacy.result_transform: "anonymize"` for the equivalent lower-level form.
When it is false or omitted, the tool result is not anonymized merely because
user/model input anonymization is enabled for the agent. A profile with
`result_policy.scan_result=true` requires scanner guardrails to be enabled.

`artifact_creator_agent_cli.py` provides a manual CLI for exercising the guarded
agent path with persistent SQLite checkpoints.

Manual CLI validation on 2026-05-18 confirmed that an artifact response
containing `https://bad.example/path` produced a `url_policy` audit decision in
`logs/artifact_creator_agent_202605181222_guardrails.jsonl`. The decision used
`mode: audit`, `rule: blocked_domain`, `boundary: model_response`, and
`source_url: true`; audit mode allowed the response while recording the
deterministic URL-policy finding.

### Finalization Handling

The confirmation path returns structured `UserConfirmation(False)` when scanner
middleware blocks the confirmation model call. The graph loops back to drafting
instead of crashing.

`final_print_node` and `ready_node` handle missing `final_artifact_url` by using
the artifact storage error message instead of raising `KeyError`.

### Transforming Any Agent For Phase 3A Guardrails

The Phase 3A transformation moves tool execution from "tools are merely passed
to the agent" to "every runtime tool has an explicit security and privacy
profile". The required outcome is that unprofiled or disallowed tools fail
closed, external access is controlled by runtime context, and tool results are
marked as trusted or untrusted before they re-enter model context.

For a platform-compiled migration, tools should be declared in the registry
`tools` block and the agent graph should use `add_agent_node(...,
tools_source="platform")`. When the policy has
`guardrail_tool_execution_enabled=True`, `bot_service.agent_registry` asks
`build_agent_tool_bundle(..., require_guardrail_profiles=True)` to resolve both
tool objects and runtime-name profiles. `PlatformGraphCompiler` then injects
`ToolExecutionSafetyMiddleware` into guarded agent nodes using the same
`PlatformGuardrailRuntime` that owns scanner and privacy rails.

For an agent-owned migration, keep accepting `tools` in `initialize_agent(...)`
and add:

- `guardrail_tool_execution_enabled`
- `guardrail_tool_profiles`
- `guardrail_unprofiled_tools`

The agent should build a `GuardedToolRegistry`, register the concrete runtime
tool objects with the resolved profiles, and add the returned
`ToolExecutionSafetyMiddleware` to the LangChain middleware list. A tool profile
that scans results requires scanner guardrails to be enabled; a profile that
anonymizes results requires privacy guardrails to be enabled. Startup should
raise rather than silently weakening those dependencies.

Tool profiles should live in `platform_tools/tools.json`. Internal tools use
`guardrail_profile`; MCP server templates use `guardrail_profiles` keyed by the
runtime tool name, with unprefixed MCP tool names accepted when applicable.
Profile `category` is the external-access control point: tools with
`category: "external_access"` require runtime
`allow_external_tool_access=True`. File export, sensitive-data, approval, and
role policies should be represented in the profile rather than in ad hoc agent
conditionals.

Phase 3A migration is complete for an agent when:

- every configured runtime tool resolves a valid guardrail profile when tool
  execution guardrails are enabled;
- unprofiled tools are blocked unless the agent explicitly accepts the
  `allow_read_only` fallback policy;
- external tools are blocked unless the runtime context allows external access;
- tool arguments are de-anonymized before execution only through the shared
  `PrivacyRail`;
- untrusted tool results are scanned or anonymized according to the tool
  profile before becoming model-visible;
- tests cover profile resolution, unprofiled-tool behavior, external-access
  gating, sensitive/file-export permissions, result anonymization, result
  scanning, and MCP startup failure handling.

## Verified Test Coverage

Focused regression coverage includes:

- scanner profile rejects `Anonymize` and `Deanonymize`;
- prompt injection blocks before model handler;
- blocked id-less messages are removed from compiled graph state;
- secrets scanner redacts and continues;
- token limit blocks;
- user-facing output does not mask Palimpsest-restored data;
- source URL memory allows source URLs but still blocks generated malicious URLs
  when the scanner classifies them as malicious;
- deterministic URL/domain policy audit and enforce behavior;
- URL-policy denylist, allowlist, private host, userinfo URL, IDNA, mixed-script,
  Unicode DNS dot separator, non-Latin configured domain, and lookalike checks;
- calibrated Russian/mixed Russian-English prompt-injection threshold is
  resolved from guardrail policy as deployment configuration;
- fail-open/fail-closed behavior;
- audit logs contain scanner metadata but no raw text;
- guardrail policy config resolves into agent initialization kwargs;
- platform tool registry resolves runtime-name guardrail profiles from
  `platform_tools/tools.json`;
- MCP connection failures are wrapped with server/URL context;
- external tool access is gated by `category == "external_access"` and runtime
  `allow_external_tool_access`;
- artifact creator middleware wiring and ordering;
- confirmation block returns `UserConfirmation(False)`;
- finalization does not crash when `final_artifact_url` is missing;
- `set_prompt` blocks hostile setup text before `state.system_prompt` is stored;
- composite scanning blocks fragmented multi-turn prompt-injection attempts
  before the model call;
- CLI helper behavior.

Representative URL-policy verification command:

```powershell
uv run pytest -q tests\unit\test_platform_guardrails_url_policy.py tests\unit\test_platform_guardrails_scanners.py
```

Last focused URL-policy/scanner run in this workstream: `48 passed`.

Last full unit run in this workstream:

```powershell
uv run pytest -q tests\unit
```

Result: `368 passed, 7 skipped`.

## Known Residual Gaps

These are outside the completed Phase 2 platform-wiring scope:

- production blocklist/allowlist contents and update process; postponed;
- public feed ingestion from sources such as URLhaus, Spamhaus, PhishTank, or
  OpenPhish;
- runtime reputation lookups such as Google Safe Browsing/Web Risk; postponed;
- URL-policy audit-to-enforce promotion; postponed until audit logs show an
  acceptable false-positive rate;
- URL-policy monitoring by structured rule/mode/boundary metrics; postponed;
- strict heuristic indirect-prompt-injection policy on top of LLM Guard;
- ongoing Russian and mixed-language scanner regression additions as new misses
  are discovered;
- fine-tuned or replacement LLM Guard-compatible models for Russian prompt
  injection, toxicity, and banned-topic handling if future evaluation shows
  recall is insufficient;
- retrieval chunk scanning for KB, tickets, and web pages;
- platform-wide rollout of role-based tool authorization beyond guarded agent
  paths;
- SQL validation;
- full policy-as-code;
- grounding checks;
- rollout of the common middleware to all agents.

The password-reset URL example demonstrates this distinction: if no deterministic
URL policy rule matches and `MaliciousURLs` scores the generated URL below the
configured threshold, the current scanner policy allows it. That is a
policy/model/list-coverage gap, not a Phase 2 wiring failure.

Russian and mixed Russian/English prompt-injection benchmarking and threshold
calibration are complete for the current workstream. The residual risk is now
operational: keep adding regression cases for newly discovered false negatives,
and revisit model fine-tuning or replacement only if future evaluation data
shows the calibrated classifier is not meeting recall targets.
