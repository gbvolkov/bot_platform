---
name: build-tco-case
description: Build a GAZ ownership-cost, operating-cost, fuel, service, maintenance, or economic-justification case. Use when the user asks about TCO, cost per km, service cost, fuel choice, ownership economics, or how to justify price through operating value.
---

# Build TCO Case

Use this skill when the user needs an ownership or operating economics argument.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- model/family/candidates or business task;
- ownership period;
- annual mileage or route;
- fuel and service assumptions;
- competitor or alternative if relevant;
- requested level: explanation, comparison, or calculation.

## What to do

1. Use `marketing_analyst` for internal TCO, fuel, service, and operations argumentation.
2. Use `gaz_pricing_bi_int` for available exact price, warranty, service interval, service-cost, and ownership-cost fields.
3. Use web only for current external assumptions if the user asks for them.
4. Do not calculate exact TCO from scratch when source inputs are missing. State assumptions and missing data.

## What to analyze

Check:
- which cost drivers are confirmed;
- which are assumptions;
- whether BI has ownership-cost fields;
- whether the requested calculation needs external current rates.

## Materials and tools

Use `task`:
- `marketing_analyst`: TCO And Operations Argumentation, Configuration Scenario Framing, Claim Safety Check.
- `gaz_pricing_bi_int`: Ownership Cost Lookup, Service And Warranty Lookup, Price Lookup, Data Availability Check.
- `web_search_agent`: Fresh Market Context when external current rates are required.

## Output

Return:
- TCO argument or comparison;
- confirmed BI facts;
- internal-material rationale;
- assumptions and missing inputs.
