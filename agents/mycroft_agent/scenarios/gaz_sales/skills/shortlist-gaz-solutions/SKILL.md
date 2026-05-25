---
name: shortlist-gaz-solutions
description: Build an initial shortlist of GAZ vehicle families, model directions, body directions, special-body directions, fuel directions, or passenger-transport directions for a customer task. Use before exact BI validation when the user needs to know what may fit.
---

# Shortlist GAZ Solutions

Use this skill when the user asks what GAZ options may fit a business task and the answer needs internal-material grounding.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- customer task and operating context;
- known requirements from `capture-customer-requirements`;
- brand/scope restrictions;
- body, route, cargo/passenger, fuel, or special-equipment constraints.

## What to do

1. Do not choose GAZ candidates from memory.
2. Ask `marketing_analyst` for internal-material candidate directions.
3. Pass the customer task neutrally. Do not seed example models unless the user named them.
4. If the user provided formal filter parameters, optionally ask `gaz_pricing_bi_int` in analytical / selected-field mode for formal GAZ catalog filtering.
5. Keep the shortlist preliminary until BI validates exact facts.
6. Do not request complete DB model profiles during initial shortlisting. Save complete profile requests for concrete model-detail turns.

## What to analyze

Check:
- whether candidates are in GAZ scope;
- whether marketing output includes internal evidence;
- which candidate facts require BI validation;
- whether BI filtering is needed now or later;
- whether the user has shifted from shortlisting to concrete model-detail retrieval.

## Materials and tools

Use `task`:
- `marketing_analyst` services: GAZ Family Overview, Special Body Application Framing, Passenger Transport Fit, Configuration Scenario Framing, TCO And Operations Argumentation.
- `gaz_pricing_bi_int` services when formal parameters exist: GAZ Catalog Filter, Model Identity Lookup, Data Availability Check.

## Output

Return a shortlist with:
- candidate family/model direction;
- why it may fit;
- what is confirmed by internal materials;
- what BI must validate before final recommendation.
