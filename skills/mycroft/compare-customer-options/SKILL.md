---
name: compare-customer-options
description: Compare GAZ models, families, configurations, body types, fuels, financing paths, or purchase options for a customer. Use when the user asks "compare", "what is better", "which option fits", or asks for a table plus recommendation.
---

# Compare Customer Options

Use this skill when the user wants a comparison rather than a single lookup.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- options to compare, or criteria for discovering options;
- customer task and constraints;
- comparison dimensions such as price, payload, body, route, service, TCO, financing, comfort, or sales argument.

## What to do

1. Split the comparison into factual and interpretive parts.
2. Use BI analytical / comparison mode for factual tables, model-to-model comparison, ranking, price/ownership/service analytics, and exact fields needed by the comparison.
3. In analytical / comparison mode, do not ask BI for a complete model profile and do not require BI to return all DB fields. BI is expected to return only selected comparison columns, candidate rows, aggregates, and gaps relevant to the comparison.
4. If the user asks for full details, full characteristics, all fields, all options, or a complete BI card for one concrete option, switch out of this skill's analytical mode and use `validate-vehicle-facts` in complete DB model profile mode.
5. If a previous BI analytical output omitted a BI-owned field needed for the comparison, make a selected-field BI follow-up for that field. If the user is now asking for the concrete model's details, request the Complete DB Model Profile instead.
6. Use marketing for customer-fit interpretation and sales narrative.
7. Use web only for current external or competitor facts.
8. Present conclusions after the evidence, not instead of evidence.

## What to analyze

Check:
- whether all options are in scope;
- whether exact fields were confirmed by BI;
- whether the BI request stayed in analytical / comparison mode and did not require complete profiles;
- whether omitted needed comparison fields require a selected-field follow-up;
- whether a user request for full details should be routed to complete DB model profile mode instead;
- whether marketing claims are clearly not presented as BI facts;
- whether the winning criterion is explicit.

## Materials and tools

Use `task`:
- `gaz_pricing_bi_int`: Candidate Comparison Table, Competitor Fact Comparison, GAZ Catalog Filter, Price Lookup, Technical Specs Lookup, Options And Equipment Lookup, Body And Platform Lookup, Ownership Cost Lookup, Service And Warranty Lookup.
- `marketing_analyst`: GAZ Model Sales Comparison, Configuration Scenario Framing, Customer Sales Arguments.
- `web_search_agent`: external/current facts when requested.

## Output

Return a comparison with:
- factual table if useful;
- customer-fit interpretation;
- recommended option or decision rule;
- caveats and missing fields.
