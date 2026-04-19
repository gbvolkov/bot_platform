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
2. Use BI for factual tables and exact fields.
3. Use marketing for customer-fit interpretation and sales narrative.
4. Use web only for current external or competitor facts.
5. Present conclusions after the evidence, not instead of evidence.

## What to analyze

Check:
- whether all options are in scope;
- whether exact fields were confirmed by BI;
- whether marketing claims are clearly not presented as BI facts;
- whether the winning criterion is explicit.

## Materials and tools

Use `task`:
- `gaz_pricing_bi_int`: Candidate Comparison Table, Price Lookup, Technical Specs Lookup, Options And Equipment Lookup, Body And Platform Lookup.
- `marketing_analyst`: GAZ Model Sales Comparison, Configuration Scenario Framing, Customer Sales Arguments.
- `web_search_agent`: external/current facts when requested.

## Output

Return a comparison with:
- factual table if useful;
- customer-fit interpretation;
- recommended option or decision rule;
- caveats and missing fields.
