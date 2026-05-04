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
3. Use `gaz_pricing_bi_int` analytical / comparison mode for formal filters, candidate rows, factual comparison tables, exact fields needed by the recommendation, and ownership/service analytics.
4. In recommendation work, do not require BI to return complete model profiles or all DB fields unless the user explicitly asks for full concrete model details. BI recommendation support may return only selected fields relevant to the recommendation.
5. If marketing returns candidates and the final answer needs prices, TTX, configurations, service, warranty, ownership facts, dimensions, geometry, body/platform, cargo/loading, or options, call BI before finalizing.
6. If BI returns fewer candidates than marketing and the gap matters, make a corrected BI follow-up request.
7. If BI analytical output omitted a field needed for the recommendation, make a selected-field BI follow-up. If the user asks for a concrete model's full details after the recommendation, use `validate-vehicle-facts` in complete DB model profile mode.
8. Keep fleet ratios meaningful. Do not split the same model/modification into two procurement quantities.

## What to analyze

Check:
- whether the recommendation is GAZ-only if requested;
- whether each recommended candidate has enough factual support;
- whether BI analytical / comparison output contains enough selected facts for the recommendation;
- whether any needed recommendation field was omitted and requires a selected-field BI follow-up;
- whether the user has shifted from recommendation to concrete model-detail retrieval;
- whether the mix reflects route, load, maneuverability, body type, and quantity;
- whether the recommendation changed after new constraints.

## Materials and tools

Use `task`:
- `marketing_analyst`: GAZ Family Overview, Customer Sales Arguments, Configuration Scenario Framing, Special Body Application Framing, Passenger Transport Fit, TCO And Operations Argumentation.
- `gaz_pricing_bi_int`: GAZ Catalog Filter, Model Identity Lookup, Price Lookup, Technical Specs Lookup, Body And Platform Lookup, Candidate Comparison Table, Ownership Cost Lookup, Service And Warranty Lookup.
- `web_search_agent` only for current public or external competitor context.

## Output

Return:
- primary recommendation;
- optional alternatives;
- BI-confirmed facts;
- marketing-backed rationale;
- caveats and one next question if needed.
