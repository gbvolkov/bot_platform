---
name: prepare-sales-argumentation
description: Prepare customer-facing GAZ sales arguments, objection handling, value propositions, safe claims, or short sales wording. Use for "how to explain", "what to say to the client", "handle objection", "prepare talking points", or "draft customer wording".
---

# Prepare Sales Argumentation

Use this skill when the task is persuasion or explanation, not exact lookup.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Direct tools: `../references/tools-and-actions.md`

## Input

Receive:
- model/family/task;
- customer profile;
- objection or sales goal;
- desired format such as bullets, script, email fragment, or presentation text.

## What to do

1. Ask `marketing_analyst` for document-backed arguments and safe wording.
2. Identify exact claims inside the argument.
3. Validate exact claims through BI selected-field mode if they will be stated as facts.
4. Do not request complete DB model profiles for sales wording unless the user asks to include a complete concrete model card.
5. Mark unsupported or weak claims instead of strengthening them.

## What to analyze

Check:
- whether the argument is supported by internal materials;
- whether it contains factual claims requiring BI;
- whether selected-field validation is enough;
- whether wording overpromises price, discount, availability, service, or financing.

## Materials and tools

Use `task`:
- `marketing_analyst`: Customer Sales Arguments, Claim Safety Check, Customer Text Drafting, Competitor Positioning Narrative.
- `gaz_pricing_bi_int`: Price Lookup, Technical Specs Lookup, Options And Equipment Lookup, Service And Warranty Lookup, Data Availability Check.

Use Gmail tools only if the user asks for email draft/send.

## Output

Return:
- concise sales arguments;
- safe customer wording;
- caveats;
- facts that need or received BI confirmation.
