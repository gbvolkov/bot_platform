# new_theodor_agent - streaming artifact-first + deterministic persistence (best-practice plan)

Date: 2026-01-30
Status: Recommended implementation plan (agent-local only, no code shown)

## Goals

* Stream artifact text to the user as it is generated (avoid long silent wait).
* Stream discussion/questions after the artifact.
* Deterministic extraction + persistence (no heuristic parsing, no tool calls for options/draft).
* Never leak service tags/structure to the client.
* Save artifacts only after user confirmation.

## Hard constraints (must hold)

* Streaming ON.
* No tool calls for options/draft.
* No platform-level filtering.
* No LLM wrappers.
* No heuristic fallback parsing beyond an explicit contract.

---

## Best-practice approach (what changes vs your draft)

### Principle 1 - Stream presentation, validate structure server-side

* The browser should receive UI-friendly deltas (artifact text and user text), not raw model framing (tags or JSON docs).
* Use SSE/custom events to carry deltas; data can be JSON, but the UI renders only data.text. SSE supports named events and metadata fields like id and retry. ([MDN Web Docs][1])

### Principle 2 - Avoid partial JSON document streaming

* Streaming a single JSON document is fragile (incomplete objects, escaping issues). If you use JSON on the wire, do it as framed records per event (one JSON object per SSE event). ([MDN Web Docs][1])

### Principle 3 - Harden delimiting against spoofing

* If you use tags in the model output, use per-request nonce tags and sanitize inputs so user content cannot accidentally trigger your parser/stream state machine. This follows OWASP guidance: treat external data as untrusted, add clear boundaries/delimiters, sanitize before inclusion. ([OWASP Cheat Sheet Series][2])

### Principle 4 - Determinism comes from validation + retry (not temperature=0)

* Use strict validation; on contract violation, auto-retry once with a format-only corrective prompt, then fail gracefully (no heuristic repair). (Standard validate-output-then-retry practice.) ([OWASP Cheat Sheet Series][2])

---

## Architecture overview

### 1) Transport contract (Server -> Client): SSE custom events

Send only these events for new_theodor_agent:

* artifact_delta: { "text": "<chunk>" }
* user_delta: { "text": "<chunk>" }
* done: { "ok": true, "parse_ok": true/false }
* Optional: ping heartbeat to keep idle connections alive (recommended; some clients/middleboxes drop idle SSE). ([MDN Web Docs][1])

SSE details (best practice):

* Content-Type text/event-stream
* Use event: to name events, data: for payload, blank line terminates an event. ([MDN Web Docs][1])
* Use id: on deltas to support reconnect/resume; use retry: to advise reconnection delay. ([html.spec.whatwg.org][3])

### 2) Model output contract (LLM -> Server): nonce-tagged dual blocks

Strict prompt contract (single response, artifact first):

```
[ARTIFACT:<NONCE>]
(clean artifact text only)
[/ARTIFACT:<NONCE>]
[USER:<NONCE>]
(user-facing questions/discussion only)
[/USER:<NONCE>]
```

Hard rules:

* Exactly one ARTIFACT block and one USER block.
* No text outside blocks.
* ARTIFACT must precede USER.
* If violated: no persistence, trigger one auto-regeneration attempt (see Failure Handling).

Nonce + sanitization:

* Generate <NONCE> per request.
* Sanitize user input and any external context to remove/escape sequences that could mimic your delimiters (or rely on nonce uniqueness to make collisions practically impossible).
* This prevents delimiter spoofing and accidental truncation. ([OWASP Cheat Sheet Series][2])

### 3) Agent-local streaming filter (callback handler)

Implement a FilteredStreamCallbackHandler that:

* Maintains a bounded buffer for cross-token delimiter detection.
* Tracks mode: OUTSIDE | IN_ARTIFACT | IN_USER.
* Drops the nonce-tag markers, never forwards them.
* Emits:
  * artifact_delta when IN_ARTIFACT
  * user_delta when IN_USER
  * nothing when OUTSIDE

This guarantees: the client sees artifact text first (as generated), then discussion text - no tags, no structure leaks.

### 4) Deterministic parsing and persistence (strict)

After model completion:

* Parse the final full text strictly using the nonce tags.
* If valid:
  * artifact_draft_text = ARTIFACT block
  * user_text = USER block
  * Persist draft only in state; emit done {parse_ok:true}.
* If invalid:
  * Run one auto-regeneration attempt with a short corrective system message ("Return only in the required format, no extra text.").
  * If still invalid: emit done {parse_ok:false}, show a user-visible error, keep last good state.

### 5) Confirmation gating (unchanged)

* On user confirmation:
  * artifact_final_text = artifact_draft_text (pure state copy).
* On change request:
  * return to generation (still no tools in draft/options steps).

---

## Optional best-practice upgrade: Structured Outputs for persistence (without showing JSON)

If your model/API supports Structured Outputs (JSON Schema), use it only at the persistence boundary, not as UI output:

* Stream UI via artifact_delta/user_delta as above.
* At completion (or after confirmation), validate/store against a schema:

```json
{ "artifact_text": "string", "user_text": "string" }
```

Structured Outputs are designed to ensure the model adheres to a supplied JSON Schema, reducing parsing errors. ([OpenAI Platform][4])

Two safe deployment variants:

* Single-call: Keep nonce tags for streaming, then store as-is (strict parser).
* Two-pass (recommended if you want schema guarantees without streaming-JSON complexity):
  1. Stream generation using nonce tags + callback filter (fast UX).
  2. Non-streamed packaging call that converts generated texts into schema-valid structured output for storage (cheap, deterministic).
     This keeps JSON entirely off the client path.

---

## Streaming modes and client handoff

### Server config

* For new_theodor_agent, stream only:
  * custom (your SSE events) and values (internal).
* Disable raw messages streaming to avoid token leakage.

### Client behavior

* Subscribe to:
  * artifact_delta -> append to Artifact panel
  * user_delta -> append to Discussion panel
  * done -> finalize UI state (enable confirm controls, show error banner if parse_ok=false)

Named SSE events are standard; EventSource.addEventListener("<event>", ...) is the canonical pattern. ([MDN Web Docs][1])

---

## Failure handling (no heuristic repair)

* Parse failure -> auto-regenerate once (format-only).
* Second failure -> user-visible error and no state mutation.
* Always emit done so the client can stop loading cleanly.

---

## Security & robustness checklist

* Delimiter hardening: nonce tags + input sanitization. ([OWASP Cheat Sheet Series][2])
* No data outside blocks (prevents prompt leakage/preamble).
* SSE robustness: include id and optionally retry; consider heartbeat to prevent idle disconnects. ([html.spec.whatwg.org][3])
* Logging/metrics: contract violation rate, retry rate, average artifact latency, token counts.
* Streaming caveat: if you have policy/moderation needs, streaming emits partial content before full validation - decide your risk posture accordingly (e.g., lightweight early checks, or restricted contexts). ([OpenAI Platform][5])

---

## Implementation steps (precise)

1) Prompts

* Add nonce-tagged strict contract (artifact first).
* Explicitly forbid tool calls for options/draft.

2) Streaming modes

* streaming.modes = ["custom", "values"] for new_theodor_agent (remove "messages").

3) Callback handler

* Implement DFA + bounded buffer.
* Emit artifact_delta and user_delta.

4) Strict parser

* Parse final full output using nonce tags.
* Validate invariants: both blocks present, correct order, nothing outside blocks.

5) Retry policy

* On parse fail: auto-retry once.
* On second fail: surface error; do not write artifact fields.

6) State updates

* Store artifact_draft_text, user_text.
* Confirmation copies draft -> final.

7) SSE niceties

* Add done event.
* Add id (monotonic per stream) and optional retry.
* Optional ping heartbeats.

---

## Verification checklist

* Streaming:
  * Client receives artifact text first (artifact_delta), then user discussion (user_delta).
  * No tags/nonce markers ever appear in streamed text.
* Parsing:
  * artifact_draft_text contains only artifact content.
  * user_text contains only discussion content.
* Confirmation:
  * artifact_final_text only written on explicit confirm.
* Reliability:
  * Reconnect behavior works with id (optional but recommended).
  * Parse failures trigger exactly one auto-retry; no heuristic repairs.

---

## Resolved open questions

* Can the client consume custom stream events? Yes; SSE supports named events and client listeners. ([MDN Web Docs][1])
* User-visible errors on contract violation? Acceptable only after one auto-retry; otherwise feels flaky.
* Regeneration auto-trigger vs user action? Auto-trigger exactly once, then surface error.

If you want, paste your current load.json plus a snippet of your existing stream writer payload shape, and I will align the event names/fields so it matches your client expectations (still within agent-local-only constraints).

[1]: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events?utm_source=chatgpt.com "Using server-sent events - Web APIs | MDN"
[2]: https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html?utm_source=chatgpt.com "AI Agent Security - OWASP Cheat Sheet Series"
[3]: https://html.spec.whatwg.org/multipage/server-sent-events.html?utm_source=chatgpt.com "9.2 Server-sent events - HTML Standard - WhatWG"
[4]: https://platform.openai.com/docs/guides/structured-outputs?utm_source=chatgpt.com "Structured model outputs | OpenAI API"
[5]: https://platform.openai.com/docs/guides/streaming-responses?utm_source=chatgpt.com "Streaming API responses"
