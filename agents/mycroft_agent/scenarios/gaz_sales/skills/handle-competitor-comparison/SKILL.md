---
name: handle-competitor-comparison
description: Handle GAZ versus competitor comparisons, competitor objections, external alternatives, and current public competitor context. Use when the user names a competitor, asks why GAZ is better/worse, requests market comparison, or needs sources about competitor offers.
---

# Handle Competitor Comparison

Use this skill when a competitor appears in the user's task.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- GAZ model/family or customer task;
- competitor brand/model if named;
- comparison criteria;
- whether the user needs internal positioning, exact facts, or current public context.

## What to do

1. Use `marketing_analyst` for internal GAZ positioning against the competitor or competitor class.
2. Use `gaz_pricing_bi_int` analytical / comparison mode when concrete competitor records or GAZ records must be compared by structured facts.
3. Use `web_search_agent` for current public competitor facts and links.
4. Do not ask BI for complete DB model profiles for competitor comparison unless the user asks for full concrete details of one model.
5. Keep internal, BI, and web conclusions separate.

## What to analyze

Check:
- whether the competitor is explicitly in scope;
- whether competitor facts are current enough;
- whether BI has competitor records;
- whether selected comparison fields are enough and complete profile mode is unnecessary;
- whether marketing arguments are supported by materials.

## Materials and tools

Use `task`:
- `marketing_analyst`: Competitor Positioning Narrative, Customer Sales Arguments, Claim Safety Check.
- `gaz_pricing_bi_int`: Competitor Fact Comparison, Candidate Comparison Table, Price Lookup, Technical Specs Lookup.
- `web_search_agent`: Competitor Public Context, Current Public Fact Lookup, External Source Links.

## Output

Return:
- GAZ position;
- factual comparison where available;
- current public context with links if used;
- risks and unconfirmed areas.
