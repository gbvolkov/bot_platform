---
name: source-authority-and-routing
description: Decide which configured Mycroft GAZ subagent or direct tool should answer a sales request. Use for recommendations, comparisons, exact vehicle facts, internal sales arguments, programs/financing, TCO, service/operation questions, current external facts, and mixed requests requiring several sources.
---

# Source Authority And Routing

Use this skill before delegating work when the current turn requires source selection.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Direct tools: `../references/tools-and-actions.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- the user's current question;
- relevant conversation history;
- known customer constraints;
- existing candidate models/families;
- previous marketing, BI, or web results.

## What to decide

Decide which source owns each part of the answer:
- `marketing_analyst` owns internal GAZ materials, sales fit, positioning, objections, programs/financing materials, TCO narrative, special-body framing, passenger-route framing, and broad competitor narrative.
- `gaz_pricing_bi_int` owns exact prices, exact structured TTX, formal model lookup, formal catalog filtering, options/equipment fields, body/platform fields, service/warranty facts, ownership-cost fields, and factual comparison tables.
- `web_search_agent` owns current public facts, source links, external verification, public competitor context, and fresh market context.
- Direct tools own explicit actions such as email draft/send or artifact export.

## Routing workflow

1. Classify the user request by claim type: broad selection, exact fact, sales argument, financing/program, TCO, service/operation, competitor, current public context, email, export.
2. Preserve the user's scope. If the user asks for GAZ-only, pass `GAZ-only; exclude non-GAZ and competitor models from the candidate list` to relevant subagents.
3. For broad fit or initial candidate discovery, use `marketing_analyst` first unless the user already gave formal filter parameters suitable for BI.
4. For exact facts about concrete models, families, candidates, or formal filters, use `gaz_pricing_bi_int`.
5. For current external facts or public links, use `web_search_agent`.
6. For mixed requests, call multiple sources and keep their outputs separated until synthesis.
7. If one source returns a gap, use the next appropriate source instead of stopping prematurely.

## BI delegation rules

Call `gaz_pricing_bi_int` only when the request contains at least one concrete BI target:
- a named model, modification, family, or candidate set;
- a formal GAZ-only filter such as body type, payload, passenger capacity, dimensions, fuel, drive, or price;
- an exact BI-owned attribute to retrieve;
- a concrete competitor fact comparison target.

When asking BI to restrict by manufacturer, use the structured field `manufacturer_brand_lat`.
The value must be Latin uppercase, for example `manufacturer_brand_lat=GAZ`.
For GAZ-only requests, explicitly pass `manufacturer_brand_lat=GAZ`.
For competitor comparison, pass the competitor manufacturer as `manufacturer_brand_lat=<UPPERCASE_LATIN_BRAND>` when it is one of the database manufacturers.

The BI database contains vehicles only for these `manufacturer_brand_lat` values:
`AVIOR`, `DONGFENG`, `FOTON`, `GAZ`, `ISUZU`, `JAC`, `KAMAZ`, `LADA`, `LIAZ`, `MAZ`, `PAZ`, `PROMTEKH`, `SAZ`, `SIMAZ`, `SOLLERS`, `UAZ`, `VOLGABUS`, `YUTONG`.

Do not ask BI to:
- create questionnaires;
- plan the sales dialogue;
- invent best options;
- write sales arguments;
- decide customer fit without formal criteria.

## Marketing delegation rules

Call `marketing_analyst` when the request needs:
- internal materials;
- broad model/family overview;
- sales fit and use-case reasoning;
- objections and value framing;
- programs, financing, discount/offer conditions from materials;
- TCO/service/operations arguments;
- special-body, fuel, configuration, or passenger-route framing.

Do not ask marketing to confirm exact price, exact payload, exact dimensions, exact option availability, exact service interval, or fresh public competitor facts.

## Web delegation rules

Call `web_search_agent` when:
- the user asks for sources/links;
- the answer needs current public information;
- competitor facts are not available internally or need freshness;
- public financing/program context must be checked.

Do not use web as the first source for internal GAZ facts or BI-owned facts.

## Output

Produce an internal source plan:
- which subagent/tool to call;
- what exact question to ask;
- which facts each source is expected to provide;
- what must be checked again if results are incomplete.
