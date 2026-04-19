---
name: email-followup
description: Prepare, draft, or send GAZ sales follow-up emails, quote summaries, recap emails, or customer messages. Use when the user asks for an email draft, customer recap, commercial follow-up, or explicit send action.
---

# Email Follow-Up

Use this skill only when the user asks for an email-related action.

Reference documents:
- Direct tools: `../references/tools-and-actions.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Input

Receive:
- recipient and subject if known;
- target customer or audience;
- recommendation, facts, caveats, and desired tone;
- whether the user wants a draft or immediate send.

## What to do

1. Identify whether the user wants text in chat, a Gmail draft, or actual sending.
2. If exact prices, TTX, options, or service facts will appear in the email, ensure they are BI-confirmed.
3. If the email needs sales wording or objection handling, ask `marketing_analyst` for safe customer-facing text.
4. Draft by default.
5. Send only when the user explicitly says to send.

## What to analyze

Check:
- whether recipient, subject, and body are sufficient for sending;
- whether any promise of price, availability, discount, financing, or service fact lacks confirmation;
- whether caveats should remain visible in the email.

## Materials and tools

Use:
- `gmail_create_draft` for drafts;
- `gmail_send_message` for explicit send requests;
- `marketing_analyst` for customer-facing wording;
- `gaz_pricing_bi_int` for exact factual validation before sending.

## Output

For chat-only requests, return the email text.
For draft requests, create a draft and summarize what was drafted.
For send requests, use the send tool only after required details are present and runtime approval permits it.
