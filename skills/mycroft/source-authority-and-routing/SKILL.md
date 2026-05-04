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
- `gaz_pricing_bi_int` owns exact prices, exact structured TTX, exact vehicle dimensions and geometry, formal model lookup, formal catalog filtering, options/equipment fields, body/platform fields, cargo/platform/loading fields, service/warranty facts, ownership-cost fields, and factual comparison tables.
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

For follow-up turns, a concrete BI target may be composed from:
- the exact attribute requested in the current user message;
- the latest active model, comparison, candidate set, or fleet mix from conversation history.

If the user asks for an exact BI-owned attribute and that exact attribute is not already present in the visible context for the active target models, route to `gaz_pricing_bi_int`. Do not infer BI absence from a previous BI answer that did not request or return that exact field.

Treat "стоимость владения", "стоимость владения в BI", "TCO", "cost per km", "руб/км", "ownership cost", "cost of ownership", and similar ownership-economics phrases as BI-owned ownership-cost attributes when the current turn or conversation history has active concrete models, candidates, comparison rows, or a fleet mix. In that case, route to `gaz_pricing_bi_int` in analytical / selected-field mode before `marketing_analyst`, web, a calculated template, or a request for assumptions, unless the exact BI ownership-cost values are already visible for those active targets.

### BI request modes

Use exactly one BI mode per request unless the user explicitly needs both.

1. **Analytical / comparison mode**
   Use for comparing different models, catalog filtering, candidate discovery, ranking, price/ownership/service tables, or answering "which is better/cheaper" from selected criteria.
   In this mode, BI is expected to return only the fields, aggregates, candidate rows, or comparison columns needed for the analytical task. BI does not return a complete model profile in this mode. Do not require BI to return all DB fields and do not ask for a complete profile.

2. **Concrete model detail mode**
   Use when the user asks for a concrete model's details, full characteristics, full options, "полный список", "все поля", "строго из BI", a complete model card, or a short follow-up that needs model details not present in visible context.
   In this mode, ask for the Complete DB Model Profile service for each concrete model. Require BI to return the complete set of DB fields for the matched model row(s), including NULL/empty fields. Ask BI to render NULL/empty values as `NA`, keep original DB field names when possible, and not omit fields because they are outside the current topic. From the returned model row(s), Mycroft forms the complete model profile and then answers only the user's requested slice unless the user asked for the full profile.

For short or multi-attribute follow-ups such as "свесы, габариты?", "а гарантия?", "а кондиционер?", or "расходы на ТО?", inherit the latest active model, comparison, candidate set, or fleet mix. If the previous BI result was analytical/comparison mode and omitted the needed concrete details, route to BI again in concrete model detail mode before saying the field is absent.

If a Complete DB Model Profile was requested but a needed field is still missing from the returned field list, route to BI again for Specific Missing Field Recovery before answering. Pass the active model(s), the missing DB/user-facing field(s), and likely aliases or schema names when known, for example `front_overhang_mm`, `rear_overhang_mm`, "передний свес", and "задний свес". Ask BI to return the value or `NA` if the DB value is NULL. Do this before asking the user to clarify units and before saying the field is absent.

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
- TCO/service/operations arguments after BI-owned ownership/service facts have been checked for concrete active targets, or when no concrete BI target exists yet;
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
