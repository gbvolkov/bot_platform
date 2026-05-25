---
name: capture-customer-requirements
description: Extract and maintain customer requirements in a GAZ sales dialogue. Use when the customer describes a vehicle, fleet, route, cargo, passenger, service, financing, or comparison task and Mycroft must identify known inputs, assumptions, missing high-impact inputs, and the next best question.
---

# Capture Customer Requirements

Use this skill before source lookup when the customer need is still being shaped.

Reference documents:
- Full skill catalog: `../references/skill-catalog.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Input

Receive:
- the user's current message;
- conversation history;
- previous constraints and recommendations;
- customer impatience signals such as "дайте варианты" or "хватит уточнять".

## What to do

1. Extract the business task: cargo, passenger, service, special body, financing, competitor comparison, or follow-up.
2. Capture known inputs: city/region, quantity, cargo/passenger profile, body or special body, route and road constraints, load/volume/passenger capacity, fuel, drive, driver category, budget, timing, financing interest.
3. Mark assumptions that can be used for a preliminary answer.
4. Identify only the missing inputs that can materially change the recommendation.
5. Ask at most one high-impact follow-up question unless the user explicitly requests a checklist.

## What to analyze

Check whether the user has already provided enough context to proceed.
Do not block on every missing field.
If the user asks for concrete options, proceed with assumptions and validate later.

## Materials and tools

Usually no tools or subagents are needed.

Do not ask `gaz_pricing_bi_int` to define what questions to ask the customer.
Do not ask `marketing_analyst` to design the dialogue unless the user requests source-backed sales methodology.

## Output

Produce an internal requirement brief:
- known facts;
- assumptions;
- missing high-impact input;
- recommended next source or action.
