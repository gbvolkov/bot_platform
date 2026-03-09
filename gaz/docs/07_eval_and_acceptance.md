# 07. Eval Plan and Acceptance Criteria

## Purpose

The eval suite should validate:
- stage discipline
- branch routing correctness
- evidence-before-shortlist policy
- branch conflict handling
- role-aware follow-up packaging

## Required eval scenarios

### 1. City refrigerator + Atlant + price sensitivity
Input:
- city
- refrigerator
- competitor Atlant
- price important

Expected:
- conflict resolved before tool use
- configuration first if user prioritizes configuration
- comparison only after branch lock and evidence

### 2. “What is your most cost-effective option?”
Input:
- vague request
Expected:
- no product recommendation
- clarification first
- likely `tco` after more information

### 3. Service risk dominates
Input:
- customer fears downtime / bad service history
Expected:
- `service_risk`
- manuals/service docs or quality docs used
- no premature shortlist

### 4. Need package for management approval
Input:
- asks for short pack for leadership
Expected:
- `internal_approval`
- role clarified
- package has rationale per doc

### 5. Passenger route selection
Input:
- city route
- wants economics + capacity
Expected:
- `passenger_route`
- route/capacity asked before shortlist

### 6. Special body request: tow truck
Input:
- “need a tow truck, not sure on chassis”
Expected:
- `special_body`
- function-first response
- chassis only after evidence

### 7. Offroad / harsh conditions
Input:
- poor roads / severe conditions / not price-first
Expected:
- `special_conditions`
- offroad fit prioritized over price

## Mandatory assertions

### Trace-level
- no product recommendation before branch lock
- no shortlist before evidence
- conflict resolved before branch-specific pack
- no more than 1 branch-pack call per evidence cycle
- no more than 2 read_doc calls per evidence cycle

### Output-level
- responses are concise
- they remain problem-first
- they explain why the current evidence matters

## Acceptance criteria

The implementation is accepted if:
1. All 7 eval scenarios pass.
2. Parent graph compiles.
3. Evidence subgraph compiles and is reusable.
4. All policy guards are covered by tests.
5. Trace artifacts clearly show:
   - stage transitions
   - tool calls
   - branch decisions
   - shortlist gating
   - follow-up packaging

## Nice-to-have

- LangSmith trace tags by stage/branch
- snapshot tests for rendered answers
- synthetic fuzz tests for ambiguous user asks
