---
name: build-vehicle-recommendation
description: Build a concrete GAZ vehicle, configuration, or fleet recommendation from customer requirements, internal marketing-material fit, BI facts, and optional web context. Use when the user asks what to buy, which fleet mix to choose, or which model/configuration to recommend.
---

# Build Vehicle Recommendation

Use this skill when the final outcome must be a recommendation, not just a lookup.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Direct tools: `../references/tools-and-actions.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- customer requirements;
- candidate list from marketing or BI;
- exact facts from BI if already available;
- prior recommendation and latest constraints.

## What to do

1. Start from the latest customer constraints.
2. Use `marketing_analyst` for business fit and internal-material rationale.
3. Use `gaz_pricing_bi_int` for exact facts on concrete candidates or formal filter matches.
4. If marketing returns candidates and the final answer needs prices, TTX, configurations, service, or warranty, call BI before finalizing.
5. If BI returns fewer candidates than marketing and the gap matters, make a corrected BI follow-up request.
6. Keep fleet ratios meaningful. Do not split the same model/modification into two procurement quantities.

## What to analyze

Check:
- whether the recommendation is GAZ-only if requested;
- whether each recommended candidate has enough factual support;
- whether the mix reflects route, load, maneuverability, body type, and quantity;
- whether the recommendation changed after new constraints.

## Materials and tools

Use `task`:
- `marketing_analyst`: GAZ Family Overview, Customer Sales Arguments, Configuration Scenario Framing, Special Body Application Framing, Passenger Transport Fit, TCO And Operations Argumentation.
- `gaz_pricing_bi_int`: GAZ Catalog Filter, Price Lookup, Technical Specs Lookup, Body And Platform Lookup, Candidate Comparison Table.
- `web_search_agent` only for current public or external competitor context.

## Output

Return:
- primary recommendation;
- optional alternatives;
- BI-confirmed facts;
- marketing-backed rationale;
- caveats and one next question if needed.
