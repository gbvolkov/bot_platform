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
3. For concrete BI candidates, request a complete non-duplicate model field profile for each option, not a curated "important fields" subset. Complete means every user-facing original BI field available for each matched row; examples such as price, payload, body, route-relevant dimensions, service, TCO, financing, comfort, options, and platform are only a minimum checklist.
4. If a previous BI output omitted a BI-owned field needed for the comparison or a follow-up answer, make a targeted BI follow-up for the active option(s), specific missing field(s), and likely aliases/schema names before saying the fact is missing.
5. Use marketing for customer-fit interpretation and sales narrative.
6. Use web only for current external or competitor facts.
7. Present conclusions after the evidence, not instead of evidence.

## What to analyze

Check:
- whether all options are in scope;
- whether exact fields were confirmed by BI;
- whether BI returned complete non-duplicate profiles instead of only a selected table;
- whether omitted needed fields require Specific Missing Field Recovery;
- whether marketing claims are clearly not presented as BI facts;
- whether the winning criterion is explicit.

## Materials and tools

Use `task`:
- `gaz_pricing_bi_int`: Complete Model Field Profile, Specific Missing Field Recovery, Candidate Comparison Table, Price Lookup, Technical Specs Lookup, Options And Equipment Lookup, Body And Platform Lookup.
- `marketing_analyst`: GAZ Model Sales Comparison, Configuration Scenario Framing, Customer Sales Arguments.
- `web_search_agent`: external/current facts when requested.

## Output

Return a comparison with:
- factual table if useful;
- customer-fit interpretation;
- recommended option or decision rule;
- caveats and missing fields.
