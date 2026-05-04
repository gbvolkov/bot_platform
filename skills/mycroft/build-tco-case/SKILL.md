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

1. If the user asks about TCO, cost per km, ownership cost, "стоимость владения", "стоимость владения в BI", "руб/км", or a similar ownership-cost fact and the current turn or conversation history contains concrete active models, families, candidates, comparison rows, or fleet mix, call `gaz_pricing_bi_int` first in analytical / selected-field mode.
2. In that BI request, include the active target model names and ask for available ownership-cost/TCO/cost-per-km fields, price, service interval, service-cost, warranty, and the calculation conditions stored in BI such as period, mileage, fuel basis, and assumptions. Ask BI to return the value or `NA` for each requested BI field.
3. Do not replace this BI ownership-cost lookup with a generic TCO template, marketing argument, or request for customer assumptions. Use a template or assumptions only after BI has been checked for the active targets and the missing BI fields are known.
4. Use `marketing_analyst` for internal TCO, fuel, service, and operations argumentation after exact BI-owned ownership/service facts have been checked, or when there is no concrete BI target yet.
5. Use web only for current external assumptions if the user asks for them, or after BI has no exact field and the user explicitly accepts external estimation.
6. Do not request complete DB model profiles for TCO analytics unless the user also asks for a full concrete model card.
7. Do not calculate exact TCO from scratch when source inputs are missing. State assumptions and missing data.

## What to analyze

Check:
- which cost drivers are confirmed;
- which are assumptions;
- whether active concrete models, candidates, comparison rows, or a fleet mix already define the BI targets;
- whether BI has ownership-cost fields;
- whether BI returned exact ownership-cost/TCO/cost-per-km fields or `NA` for each active target before Mycroft produced a calculated template;
- whether the BI request is analytical / selected-field mode rather than complete profile mode;
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
