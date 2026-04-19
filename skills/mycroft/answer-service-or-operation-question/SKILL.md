---
name: answer-service-or-operation-question
description: Answer GAZ service, warranty, maintenance, operation, reliability, usage restriction, or service-cost questions. Use when the user asks about ТО, warranty, maintenance interval, service risk, operating conditions, or how to explain service/operation topics to a customer.
---

# Answer Service Or Operation Question

Use this skill for service and operation questions.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- model/family/candidate if known;
- service or operation question;
- customer use case;
- requested answer style: exact fact, explanation, objection handling, or customer text.

## What to do

1. Use BI for exact service interval, warranty, service-cost, and ownership-cost facts.
2. Use marketing for operations/service narrative, risk explanation, and safe customer wording.
3. If the model is unknown and exact facts are requested, ask one clarifying question or provide a conditional answer by family if the family is known.
4. Do not diagnose defects or make warranty decisions beyond the available source evidence.

## What to analyze

Check:
- whether the question is factual or explanatory;
- whether the exact model is required;
- whether BI has the field;
- whether marketing materials support the explanation.

## Materials and tools

Use `task`:
- `gaz_pricing_bi_int`: Service And Warranty Lookup, Ownership Cost Lookup, Data Availability Check.
- `marketing_analyst`: TCO And Operations Argumentation, Claim Safety Check, Customer Text Drafting.

## Output

Return:
- exact service facts if BI confirms them;
- operational explanation from materials if needed;
- caveats and missing confirmations.
