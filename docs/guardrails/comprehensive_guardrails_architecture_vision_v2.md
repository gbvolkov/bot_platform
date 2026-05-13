# Comprehensive Guardrails Layer — Target Architecture Vision

## 1. Executive Summary

This document proposes a target architecture for a comprehensive guardrails layer in `bot_platform`.

For the repository's current implementation status, see
[Guardrails Implementation Status](implementation_status.md).

The core decision for the next implementation phase is:

```text
Use ONLY Palimpsest for anonymization and de-anonymization.

Use LLM Guard for scanner orchestration and security scanning,
but NOT for Anonymize / Deanonymize in the production runtime path.
```

This means Palimpsest is the single owner of reversible privacy mappings. LLM Guard is introduced as a complementary guardrail framework for prompt injection, secrets, malicious URLs, toxicity/banned topics, token limits, scanner orchestration, and explicit sensitive-output checks at selected internal/export boundaries.

A future decision can revisit whether to keep Palimpsest, migrate anonymization to LLM Guard, or wrap Palimpsest as a custom LLM Guard scanner. That decision should be made later after benchmark testing on Russian/domain-specific data.

---

## 2. Architectural Principle

```text
LangGraph = orchestration and enforcement plane
Palimpsest = sole reversible anonymization/de-anonymization rail
LLM Guard = prompt/security scanner and scanner orchestration layer
Policy layer = role/tool/data authorization
Validation layer = output, SQL, schema, and grounding checks
Observability layer = redacted traces and guardrail audit trail
```

NeMo Guardrails can be evaluated later as an optional dialog/retrieval rail, but it should not be treated as native LangGraph middleware and should not replace the core guardrail architecture.

---

## 3. Current Decision: One Active Anonymizer

### 3.1 Decision

Use **only Palimpsest** for:

```text
anonymization
de-anonymization
reversible PII substitution
Russian/domain-specific PII handling
fuzzy restoration
```

Do **not** use LLM Guard `Anonymize` or `Deanonymize` scanners in the production runtime path while Palimpsest is active.

### 3.2 Rationale

Reversible anonymization depends on a stable mapping context. Running two independent anonymizers over the same text introduces ambiguity and unreliable restoration.

Example failure mode:

```text
Original:
  Иван Иванов, паспорт 4519345678

Palimpsest:
  Петр Сидоров, паспорт 7829341111

LLM Guard Anonymize:
  [PERSON_1], [ID_1]

Problem:
  Which mapping restores the final output?
  Which fake value belongs to which vault/context?
  What happens if one anonymizer rewrites another anonymizer's fake value?
```

Therefore, the architecture should enforce this rule:

```text
Exactly one anonymization/de-anonymization owner per runtime path.
```

For the current implementation phase, that owner is Palimpsest.

---

## 4. Role of Palimpsest

Palimpsest is the platform privacy transformer.

It should handle:

```text
RU_PERSON
RU_ADDRESS
RU_PASSPORT
PHONE_NUMBER
EMAIL_ADDRESS
CREDIT_CARD
SNILS
INN
RU_BANK_ACC
URL
IP_ADDRESS
TICKET_NUMBER
domain-specific identifiers
fuzzy de-anonymization after LLM paraphrasing
```

Palimpsest should be wrapped in a platform-level `PrivacyRail` abstraction.

```python
class PrivacyRail:
    def anonymize_text(
        self,
        text: str,
        context: GuardrailContext,
        boundary: str,
    ) -> str:
        ...

    def deanonymize_text(
        self,
        text: str,
        context: GuardrailContext,
        boundary: str,
    ) -> str:
        ...

    def reset_context(
        self,
        context: GuardrailContext,
    ) -> None:
        ...
```

Palimpsest should not be used as a general guardrail framework. It should not be responsible for:

```text
prompt injection detection
jailbreak detection
malicious URL detection
toxic or banned content detection
tool authorization
SQL validation
output grounding
business policy enforcement
observability policy
```

---

## 5. Role of LLM Guard

LLM Guard should be used for scanner orchestration and security scanning, but **not** as the anonymization/de-anonymization engine.

Use LLM Guard for:

```text
prompt injection detection
secrets detection
malicious URL detection
toxic/banned content detection
sensitive output detection at explicit internal/export boundaries
token limits
general scanner orchestration
```

Do not enable LLM Guard `Anonymize` or `Deanonymize` scanners while Palimpsest is the active privacy rail.

Recommended positioning:

```text
Input
  -> LLM Guard security scanners
  -> Palimpsest anonymization
  -> model / tools / retrieval
  -> output validation
  -> policy-gated Palimpsest de-anonymization
  -> egress safety scan without user-facing PII masking
  -> response
```

---

## 6. Current State in `bot_platform`

In `agents/sd_ass_agent/agent.py`, Palimpsest is currently used as follows:

```text
Palimpsest imported from palimpsest
instantiated when cfg.USE_ANONIMIZER is enabled
configured with run_entities
used in SDAgentAnonymizationMiddleware.modify_model_request
de-anonymized in validate_answer
```

This is a good starting point, but the implementation should be moved out of the service-desk agent and into reusable platform middleware.

Current gap:

```text
Palimpsest is used near model request/response,
but not consistently across retrieval, tools, SQL, memory, logs, and final output policy.
```

---

## 7. Main Gaps to Close

The current Palimpsest integration is partial.

```text
User input                    partially covered
Model request                 partially covered
Retrieved KB chunks           not fully covered
Ticket search results         not fully covered
Tool arguments                not fully covered
Tool outputs                  not fully covered
SQL queries                   not covered
SQL result rows               not covered
Generated files               not covered
Final answer                  partially covered
Memory/checkpoints            not covered
JSON/Langfuse logs            not safely covered
Prompt injection              not covered by Palimpsest
Tool authorization            not covered
Output grounding              not comprehensively covered
Role-based de-anonymization   not covered
```

---

## 8. Target Architecture Overview

```text
Client / Telegram / API
        |
        v
Request Normalization
        |
        v
Guardrail Ingress
  - request parsing
  - LLM Guard prompt/security scans
  - Palimpsest anonymization
  - role and risk classification
        |
        v
LangGraph Orchestration
  - routing
  - agent selection
  - graph state management
  - policy-aware transitions
        |
        v
Agent Execution
  - pre-model checks
  - model request anonymization
  - retrieval guardrails
  - tool guardrails
  - SQL guardrails
  - post-tool sanitization
        |
        v
Guardrail Egress
  - output validation
  - grounding checks
  - policy-gated Palimpsest de-anonymization
  - egress safety scan without user-facing PII masking
  - refusal / fallback / human review
        |
        v
Redacted Observability
  - safe logs
  - redacted Langfuse traces
  - guardrail decisions
  - audit trail
```

---

## 9. Proposed Package Structure

Create a reusable package:

```text
platform_guardrails/
  __init__.py
  context.py
  decisions.py
  privacy.py
  scanners.py    # Phase 2
  injection.py   # Phase 2
  retrieval.py
  tool_policy.py
  sql_policy.py
  output_policy.py
  memory_policy.py
  logging.py
  middleware.py
  nodes.py
```

| Module               | Responsibility                                                             |
| -------------------- | -------------------------------------------------------------------------- |
| `context.py`       | Build runtime guardrail context from LangGraph config and state            |
| `decisions.py`     | Shared decision/result objects                                             |
| `privacy.py`       | Palimpsest lifecycle, anonymization, de-anonymization                      |
| `scanners.py`      | LLM Guard scanner integration, introduced in Phase 2                       |
| `injection.py`     | Prompt injection, jailbreak, exfiltration detection, introduced in Phase 2 |
| `retrieval.py`     | Retrieved chunk scanning and source trust labeling                         |
| `tool_policy.py`   | Tool authorization, argument validation, post-tool scanning                |
| `sql_policy.py`    | SQL parse/validate/deny policy                                             |
| `output_policy.py` | Output safety, grounding, final PII checks                                 |
| `memory_policy.py` | Checkpoint and memory sanitization                                         |
| `logging.py`       | Redacted JSON/Langfuse tracing                                             |
| `middleware.py`    | LangChain/LangGraph middleware adapters                                    |
| `nodes.py`         | Graph-level guardrail nodes                                                |

---

## 10. Guardrail Decision Contract

All guardrail components should return a consistent decision object.

```python
from typing import Literal, TypedDict, Any

class GuardrailDecision(TypedDict):
    allowed: bool
    action: Literal[
        "allow",
        "block",
        "redact",
        "rewrite",
        "review",
        "fallback",
    ]
    reason: str
    risk_score: float
    categories: list[str]
    metadata: dict[str, Any]
```

Example categories:

```text
pii
prompt_injection
secret
malicious_url
toxic_content
banned_topic
token_limit
sql_policy
tool_policy
retrieval_trust
output_sensitivity
deanonymization_policy
```

---

## 11. Guardrail Runtime Context

Every decision should receive explicit runtime context.

```python
from typing import Literal, TypedDict

class GuardrailContext(TypedDict):
    tenant_id: str | None
    user_id: str | None
    user_role: str
    thread_id: str
    agent_name: str
    route: str | None
    model: str | None
    tool_name: str | None
    request_id: str
    risk_level: Literal["low", "medium", "high", "critical"]
    allow_deanonymization: bool
    allow_external_tool_access: bool
    allow_file_export: bool
    allow_sensitive_data: bool
```

The existing `ConfigSchema` already contains useful fields:

```text
user_id
user_role
thread_id
attachments
database_url
database_prompt_context
return_files
return_images
```

These should become first-class guardrail inputs.

---

## 12. Palimpsest Context Lifecycle

Palimpsest should be scoped by:

```text
tenant_id + user_id + thread_id
```

Avoid one global Palimpsest instance per agent. A global instance risks cross-thread mapping leakage.

Required lifecycle operations:

```text
on new thread:
  create privacy context

on model request:
  anonymize relevant content

on output:
  de-anonymize only if policy allows

on reset:
  reset Palimpsest context

on thread close / TTL expiry:
  dispose Palimpsest context
```

Privacy boundaries:

```text
user input          -> before LLM
retrieved chunks    -> before LLM
tool arguments      -> before tool execution, if needed
tool outputs        -> before model sees them
SQL result rows     -> before answer-generation LLM
memory writes       -> before checkpoint/storage
logs/traces         -> before persistence
final answer        -> before user delivery
```

---

## 13. LLM Guard Scanner Layer

In Phase 2, LLM Guard should be integrated through
`platform_guardrails/scanners.py`.

Use LLM Guard for:

```text
PromptInjection scanner
Secrets scanner
MaliciousURLs scanner
Toxicity scanner
BanTopics scanner
Sensitive output scanner
  (not in the user-facing default profile; reserve for explicit internal/export boundaries)
TokenLimit scanner
general scanner orchestration
```

Explicitly excluded while Palimpsest is active:

```text
Anonymize
Deanonymize
```

Scanner flow:

```text
input text
  -> LLM Guard non-anonymization scanners
  -> GuardrailDecision list
  -> allow / block / redact / review
```

For generated output, URL safety should distinguish origin. A suspicious URL
invented by the model can be blocked. A suspicious URL already present in the
user's source material may be preserved in a CRM/artifact record with warning
text, because the platform is recording supplied source data rather than
recommending a new link.

---

## 14. LangGraph / LangChain Middleware Layer

The current nested `SDAgentAnonymizationMiddleware` should be refactored into reusable middleware.

Target middleware chain:

```python
middleware = [
    SecurityScannerMiddleware(),        # LLM Guard scanners, no anonymization
    PrivacyModelRequestMiddleware(),    # Palimpsest anonymization
    ToolCallPolicyMiddleware(),
    OutputSafetyMiddleware(),
]
```

Target model request flow:

```text
original model request
  -> build GuardrailContext
  -> LLM Guard security scans
  -> Palimpsest anonymization
  -> record redacted guardrail event
  -> pass request to model
```

Target model response flow:

```text
raw model response
  -> output validation
  -> role/policy check
  -> Palimpsest de-anonymization if allowed
  -> LLM Guard egress safety scan without user-facing PII masking
  -> response rewrite/block/review if needed
```

---

## 15. Graph-Level Guardrail Nodes

Middleware is not sufficient for all controls. Some checks should be graph nodes.

For `sd_ass_agent`, evolve the graph from:

```text
START
  -> fetch_user_info
  -> reset_or_run
  -> augment_query
  -> route_request
  -> agent
  -> END
```

to:

```text
START
  -> fetch_user_info
  -> guard_ingress
  -> reset_or_run
      -> reset_memory
      -> augment_query
  -> guard_augmented_context
  -> route_request
      -> sd_agent
      -> sm_agent
      -> default_agent
  -> guard_egress
  -> END
```

| Node                        | Responsibility                                                  |
| --------------------------- | --------------------------------------------------------------- |
| `guard_ingress`           | Security scan, request classification, Palimpsest anonymization |
| `guard_augmented_context` | Scan glossary/retrieved context before model use                |
| `guard_tool_result`       | Sanitize tool outputs                                           |
| `guard_egress`            | Output validation, de-anonymization policy, final scan          |
| `guard_memory_write`      | Sanitize state before checkpointing                             |

---

## 16. Prompt Injection and Jailbreak Rail

Palimpsest does not solve prompt injection.

LLM Guard should detect:

```text
direct prompt injection
indirect prompt injection in retrieved documents
tool-output injection
system prompt extraction attempts
role override attempts
data exfiltration requests
jailbreak attempts
malicious instructions in web search results
```

Enforcement actions:

```text
allow      -> continue
redact     -> remove hostile span
block      -> refuse
review     -> human escalation
fallback   -> answer from trusted source only
```

---

## 17. Retrieval Guardrails

Retrieval is a major trust boundary. KB results, ticket search results, and web search results should not be passed directly to the model without checks.

Target retrieval flow:

```text
query
  -> LLM Guard scanner checks
  -> Palimpsest anonymization where needed
  -> retrieve documents
  -> source policy check
  -> per-chunk injection scan
  -> per-chunk PII/sensitivity scan
  -> trust labeling
  -> context assembly
  -> model
```

Guarded chunk schema:

```python
from typing import Literal, TypedDict

class GuardedChunk(TypedDict):
    text: str
    source: str
    source_type: str
    trust_level: Literal["trusted", "internal", "external", "untrusted"]
    contains_pii: bool
    injection_score: float
    allowed: bool
```

Source policy:

```text
Internal KB chunks:
  allowed but scanned

Ticket chunks:
  scanned for PII and access rights

External web chunks:
  treated as untrusted
  aggressively injection-scanned
  never interpreted as instructions
```

---

## 18. Tool Guardrails

Every tool should have a security profile.

```python
from typing import Literal, TypedDict

class ToolSecurityProfile(TypedDict):
    name: str
    category: Literal[
        "retrieval",
        "web_search",
        "database",
        "file_export",
        "notification",
        "write_action",
    ]
    side_effect: Literal["none", "read", "write", "external"]
    data_sensitivity: Literal[
        "public",
        "internal",
        "confidential",
        "regulated",
    ]
    requires_approval: bool
    allowed_roles: list[str]
```

Target tool execution flow:

```text
tool call requested
  -> tool allowlist check
  -> role policy check
  -> argument schema validation
  -> Palimpsest anonymization/redaction if needed
  -> human approval if required
  -> execute tool
  -> LLM Guard post-tool security scan
  -> Palimpsest result anonymization where needed
  -> result minimization
  -> return sanitized result to graph
```

---

## 19. SQL Guardrails

For BI/reporting flows, SQL safety must be deterministic. Do not rely only on prompt instructions such as “prefer SELECT”.

Target SQL flow:

```text
user question
  -> LLM Guard security scan
  -> Palimpsest anonymization
  -> SQL generation LLM
  -> SQL parser validation
  -> policy check
  -> execution
  -> result minimization
  -> Palimpsest result anonymization
  -> answer generation
  -> output validation
```

SQL rules:

```text
allow only SELECT / WITH SELECT
deny multiple statements
deny INSERT / UPDATE / DELETE / DROP / ALTER / CREATE
deny COPY / ATTACH / LOAD / INSTALL
deny filesystem/network access functions
deny access to blocked tables
deny sensitive columns unless role permits
enforce LIMIT unless aggregate query
validate rewritten SQL after retry/fix
```

Recommended dependency:

```text
sqlglot
```

---

## 20. Output Guardrails

Output validation should include more than de-anonymization.

Target output flow:

```text
raw model answer
  -> structure/schema validation
  -> grounding check
  -> unsafe content check
  -> policy check for de-anonymization
  -> Palimpsest de-anonymization if allowed
  -> LLM Guard egress safety scan without user-facing PII masking
  -> final response decision
```

Response actions:

```text
allow
redact
rewrite
fallback to trusted search
ask clarification
human review
block/refuse
```

---

## 21. Memory and Checkpoint Guardrails

LangGraph checkpointing can become a hidden PII persistence layer.

Target rule:

```text
Never checkpoint raw sensitive content unless explicitly allowed.
```

Memory write flow:

```text
state to persist
  -> remove unnecessary attachments
  -> Palimpsest anonymize sensitive message content
  -> strip transient tool outputs
  -> persist checkpoint
```

Reset flow:

```text
reset requested
  -> remove messages
  -> reset Palimpsest context for thread
  -> remove transient guardrail state
  -> remove temporary retrieved chunks
```

---

## 22. Observability and Audit Guardrails

Logging must become privacy-aware by default.

Target logging flow:

```text
raw event
  -> Palimpsest anonymization/redaction
  -> structured guardrail event
  -> safe JSON log
  -> safe Langfuse trace
```

Create:

```python
class RedactingJSONFileTracer(BaseCallbackHandler):
    ...
```

### Trusted Langfuse Deployment Decision

Langfuse may be enabled regardless of `guardrails_enabled` when the configured
Langfuse endpoint is part of the trusted platform environment. This is a
deployment trust decision, not a privacy transformation guarantee.

Required operating assumptions:

```text
Langfuse endpoint is controlled by the platform/operator
Langfuse credentials are stored only in environment configuration
Operators do not configure untrusted third-party Langfuse endpoints for guarded agents
Local JSON tracing remains redacted through RedactingJSONFileTracer
Guardrail decision logs remain structured and do not include Palimpsest mappings
```

If Langfuse is moved outside the trusted environment, it must be disabled for
guarded agents or routed through an explicit redaction callback before use.

Never log:

```text
raw user PII
raw tool outputs
raw SQL result rows with sensitive fields
token-by-token sensitive content
Palimpsest true/fake mappings
API keys or credentials
de-anonymized final answers unless explicitly allowed
```

---

## 23. Policy-as-Code Layer

Start with simple YAML policy files.

```text
policies/
  default.yaml
  service_desk.yaml
  sales_manager.yaml
```

Example:

```yaml
roles:
  service_desk:
    allow_tools:
      - search_kb
      - search_tickets
      - lookup_term
      - lookup_abbreviation
    allow_deanonymization: true
    allow_external_tool_access: true
    allow_file_export: false
    allow_sensitive_ticket_data: true

  sales_manager:
    allow_tools:
      - search_kb
      - lookup_term
      - lookup_abbreviation
    allow_deanonymization: false
    allow_external_tool_access: true
    allow_file_export: false
    allow_sensitive_ticket_data: false

  default:
    allow_tools:
      - search_kb
      - lookup_term
      - lookup_abbreviation
    allow_deanonymization: false
    allow_external_tool_access: false
    allow_file_export: false
    allow_sensitive_ticket_data: false
```

Later, move to OPA if policy must be centrally governed or audited.

---

## 24. Future Decision: Palimpsest vs LLM Guard Anonymization

The current decision is to use Palimpsest only for anonymization/de-anonymization.

Later, the platform can evaluate whether to:

```text
1. Keep Palimpsest as the long-term privacy rail
2. Replace Palimpsest with LLM Guard Anonymize/Deanonymize
3. Wrap Palimpsest as custom LLM Guard scanners
```

This decision should be based on benchmarks, not framework preference.

Benchmark corpus:

```text
RU_PERSON
RU_ADDRESS
RU_PASSPORT
PHONE_NUMBER
EMAIL_ADDRESS
CREDIT_CARD
SNILS
INN
RU_BANK_ACC
URL
IP_ADDRESS
TICKET_NUMBER
organization names
mixed Russian/English support tickets
LLM-paraphrased anonymized answers
```

Benchmark metrics:

```text
detection recall
false positives
Russian language quality
domain identifier coverage
reversible de-anonymization accuracy
fuzzy restoration after LLM paraphrasing
latency
configuration complexity
failure modes
```

Decision table:

| Decision                     | When to choose                                                 |
| ---------------------------- | -------------------------------------------------------------- |
| Keep Palimpsest              | Best Russian/domain PII quality and reliable restoration       |
| Replace with LLM Guard       | LLM Guard matches quality and reduces maintenance              |
| Wrap Palimpsest in LLM Guard | You want one scanner pipeline but need Palimpsest domain logic |

---

## 25. Implementation Roadmap

### Phase 1 — Foundation. Status: Platform primitives implemented, rollout partial

Create:

```text
platform_guardrails/context.py
platform_guardrails/decisions.py
platform_guardrails/privacy.py
platform_guardrails/logging.py
platform_guardrails/middleware.py
```

Actions:

```text
Move SDAgentAnonymizationMiddleware out of sd_ass_agent/agent.py
Introduce PrivacyRail wrapper
Scope Palimpsest by user_id + thread_id
Replace JSONFileTracer with RedactingJSONFileTracer
Add guardrail decision logging
Document trusted Langfuse observability assumption
```

Current implementation details:

```text
context.py, decisions.py, privacy.py, logging.py, and middleware.py exist
PrivacyRail and PrivacyModelRequestMiddleware are implemented
artifact_creator_agent uses the common privacy middleware when guardrails are enabled
legacy/global migration for every agent is not complete
```

### Phase 2 — Scanner Enforcement. Status: Mostly implemented for artifact_creator_agent

Create:

```text
platform_guardrails/scanners.py
platform_guardrails/injection.py
```

Add LLM Guard scanners for:

```text
prompt injection detection
secrets detection
malicious URL detection
toxic/banned content detection
sensitive output detection only for explicit internal/export boundaries
token limits
```

Actions:

```text
Run scanner layer before Palimpsest anonymization
Do not run Sensitive on default user-facing responses after de-anonymization
Block or review high-risk requests
Record scanner decisions in audit logs
```

Current implementation details:

```text
scanners.py and injection.py exist
SecurityScannerMiddleware and ToolContentScannerMiddleware are implemented
guarded_node exists for scanner/privacy enforcement around graph nodes
artifact_creator_agent is the sample guarded integration
artifact_creator_agent.set_prompt is guarded before writing user-defined prompt state
composite PromptInjection scanning covers privileged prompt state plus recent messages
scanner policy trusts LLM Guard allow/block decisions
deterministic phishing URL policy is not implemented in Phase 2
```

Current status note:

```text
Phase 2 platform wiring is mostly done for the sample integration.
The remaining Phase 2 hardening item is scanner-model quality for Russian and mixed Russian/English traffic.
The default LLM Guard models need benchmark evaluation, threshold calibration, and likely fine-tuning or replacement before broad production rollout.
This is not an invitation to add deterministic prompt-policy heuristics; LLM Guard remains the scanner decision source.
```

### Phase 2 Hardening — Russian Scanner Model Quality

Actions:

```text
Build Russian and mixed Russian/English scanner evaluation sets
Measure PromptInjection, Toxicity, BanTopics, and MaliciousURLs behavior on product-domain traffic
Calibrate scanner thresholds per deployment profile
Fine-tune or replace LLM Guard-compatible scanner models where recall is insufficient
Keep Palimpsest as the only anonymizer/de-anonymizer
Keep deterministic prompt-policy checks out of the common scanner layer unless a later policy phase explicitly adds them
```

### Phase 3 — Execution Safety

Create:

```text
platform_guardrails/tool_policy.py
platform_guardrails/sql_policy.py
platform_guardrails/retrieval.py
```

Actions:

```text
Add role-based tool access
Add read-only SQL validation
Scan retrieved KB/ticket chunks
Scan tool outputs
Minimize sensitive result rows
```

### Phase 4 — Output Safety

Create:

```text
platform_guardrails/output_policy.py
```

Actions:

```text
Add final answer validation
Add groundedness checks
Add controlled Palimpsest de-anonymization
Add explicit sensitive-data output policy for selected internal/export boundaries
Add fallback/review/block actions
```

### Phase 5 — Policy-as-Code

Create:

```text
policies/
  default.yaml
  service_desk.yaml
  sales_manager.yaml
```

Actions:

```text
Move role permissions out of code
Add policy test cases
Add audit trail for policy decisions
Optionally evaluate OPA integration
```

### Phase 6 — Future Anonymizer Evaluation

Actions:

```text
Build benchmark corpus
Compare Palimpsest vs LLM Guard Anonymize/Deanonymize
Measure Russian/domain-specific recall and de-anonymization accuracy
Decide whether to keep Palimpsest, migrate, or wrap Palimpsest as custom LLM Guard scanners
```

---

## 26. Success Criteria

The guardrails layer is successful when:

```text
Palimpsest is the only active anonymizer/de-anonymizer
LLM Guard Anonymize/Deanonymize are not active in production
LLM Guard security scanners are active
No raw PII is sent to external LLMs unless explicitly allowed
No raw PII is written to logs or traces
Palimpsest mappings are scoped per user/thread
Tool calls are role-authorized
SQL is deterministically read-only
Retrieved documents are scanned before model use
External web content is treated as untrusted
Final answers pass output validation
De-anonymization is policy-gated
Guardrail decisions are auditable
Agents reuse the same platform guardrail layer
```

---

## 27. Final Target Vision

The target architecture should be:

```text
Comprehensive guardrails as a reusable platform layer,
not as agent-specific ad hoc code.
```

Current explicit decision:

```text
Palimpsest is the sole anonymization/de-anonymization engine.
LLM Guard is the scanner and scanner-orchestration layer.
```

The complete guardrail platform should cover:

```text
PII privacy through Palimpsest
prompt injection through LLM Guard
secrets detection through LLM Guard
malicious URL detection through LLM Guard
toxic/banned content detection through LLM Guard
sensitive output detection through LLM Guard at explicit internal/export boundaries
token limits through LLM Guard
retrieval trust
tool authorization
SQL safety
output validation
controlled de-anonymization
memory protection
redacted observability
policy auditability
```

This gives `bot_platform` a guardrail architecture that is:

```text
LangGraph-compatible
open-source-first
incrementally adoptable
auditable
role-aware
privacy-preserving
suitable for multi-agent expansion
```
