---
name: validate-vehicle-facts
description: Validate exact structured vehicle facts through BI for GAZ and BI-known competitor manufacturers including DONGFENG, FOTON, ISUZU, JAC, KAMAZ, LADA, MAZ, PAZ, SOLLERS, UAZ, YUTONG. Use before web for prices, TTX, body/platform, options, service, warranty, ownership cost, and factual comparison when the brand is in BI. If BI remains incomplete after a corrected follow-up, use web only for missing public/current facts or source links. Pass manufacturer scope as manufacturer_brand_lat in Latin uppercase.
---

# Validate Vehicle Facts

Use this skill whenever an exact vehicle fact is needed for GAZ or for a competitor manufacturer present in BI.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- concrete models, families, modifications, candidates, or formal filters;
- exact attributes requested;
- customer scope such as GAZ-only or competitor comparison;
- manufacturer scope when known, expressed for BI as `manufacturer_brand_lat` in Latin uppercase;
- previous marketing candidate list if any.

## What to do

1. Build a focused BI request.
2. Include concrete model/family/candidate names when known.
3. Include formal filters when the user gave them.
4. Include manufacturer restriction through `manufacturer_brand_lat` when manufacturer scope is known. Use Latin uppercase values only, for example `manufacturer_brand_lat=GAZ`.
5. For GAZ-only validation, explicitly pass `manufacturer_brand_lat=GAZ`.
6. For competitor validation, use `manufacturer_brand_lat` only when the manufacturer is one of the known BI manufacturers: `AVIOR`, `DONGFENG`, `FOTON`, `GAZ`, `ISUZU`, `JAC`, `KAMAZ`, `LADA`, `LIAZ`, `MAZ`, `PAZ`, `PROMTEKH`, `SAZ`, `SIMAZ`, `SOLLERS`, `UAZ`, `VOLGABUS`, `YUTONG`.
7. If at least one concrete model, modification, or candidate row is in scope, request a complete non-duplicate model field profile for every concrete model. Do this even when the user asks for one field, two fields, or shorthand such as "свесы, габариты?".
8. Complete means every user-facing original BI field available for the matched model record, not a curated "important fields" subset. If you list examples such as dimensions, wheelbase, maneuverability, payload, platform, service, warranty, price, or options, state that these examples are a minimum checklist and that BI must still return all other non-duplicate user-facing fields.
9. In the BI request, ask BI to exclude duplicate mirror fields and only duplicate/non-user-facing fields: `_nocase` mirror fields, normalized copies that repeat the same user-facing value, raw/import/source technical columns, and other duplicate aliases. Keep original human-readable fields, filled option/status fields, service/warranty/ownership fields, geometry, body/platform, price, TTX, identity fields, and explicit user-relevant gaps.
10. Ask for gaps explicitly: fields present and fields missing.
11. Do not infer BI absence from prior BI output that did not request a complete non-duplicate model profile or did not return the exact field. Before saying "нет в BI", "не возвращается", or "BI does not contain this", make a focused BI request for the active model(s) and the full non-duplicate profile, unless that full profile is already present in the visible context.
12. If the current question needs a BI-owned field that was omitted from the previous BI output, immediately make a targeted BI follow-up for the active model(s) and the specific missing field(s). Include likely aliases or schema names when known, for example for overhangs ask for `front_overhang_mm`, `rear_overhang_mm`, "передний свес", and "задний свес". Do this before asking the user to clarify units or before stating the field is absent.
13. If BI returns no candidates or no records for marketing candidates, do not stop there. Make one corrected BI follow-up request using more formal lookup terms: concrete family names, model name fragments, manufacturer scope through `manufacturer_brand_lat`, body/platform type, payload or mass range, drive, fuel, price range, or other structured filters already known from the user or marketing evidence.
14. When reformulating after an empty BI result, explicitly ask BI to check whether the issue is naming, manufacturer restriction, body-type synonym, missing field coverage, or true absence of matching records.
15. If the corrected BI follow-up still lacks a required fact, decide whether the missing fact is public/current and can be checked externally. If yes, use `web_search_agent` only for that missing fact or source links.
16. Do not use web to override a BI value. Use web only to fill BI gaps, provide public links, or check current public facts.
17. Do not ask BI to choose a sales winner without formal criteria.

## Routing shortcut

If the user names a competitor manufacturer that is present in BI, use BI before web for structured vehicle facts and factual comparison.
For example, for Dongfeng use `manufacturer_brand_lat=DONGFENG` and ask BI for candidate identity, body/platform, GVW, wheelbase, payload, options, service fields, and gaps before using web for fresh public links or fields missing from BI.

## What to analyze

Check:
- whether BI returned the requested exact fields;
- whether BI returned a complete non-duplicate model profile for each concrete model in scope;
- whether the BI request accidentally limited the output to a list of "important" fields instead of all non-duplicate user-facing fields;
- whether records are in scope;
- whether manufacturer scope was passed through `manufacturer_brand_lat` when relevant;
- whether missing fields affect the answer;
- whether a previous BI result actually returned a full non-duplicate model profile or only adjacent fields for the same model;
- whether the current answer needs a field omitted from the previous BI output, requiring a targeted BI re-query for that exact field and aliases;
- whether a marketing candidate was missed due to naming/body synonym;
- whether a follow-up BI request is needed;
- whether an empty BI result has been retried with concrete families, body/platform type, and formal filters before treating the fact as unavailable;
- whether any remaining missing fact should be checked through web because it is public, current, or requires source links.

## Materials and tools

Use `task -> gaz_pricing_bi_int` first.
Use `task -> web_search_agent` only after BI and corrected BI follow-up are incomplete, and only for missing public/current facts or source links.

Useful services:
- Complete Model Field Profile;
- Specific Missing Field Recovery;
- Model Identity Lookup;
- GAZ Catalog Filter;
- Price Lookup;
- Technical Specs Lookup;
- Body And Platform Lookup;
- Options And Equipment Lookup;
- Service And Warranty Lookup;
- Ownership Cost Lookup;
- Candidate Comparison Table;
- Competitor Fact Comparison;
- Data Availability Check.

## Output

Return BI-backed facts and gaps from the complete non-duplicate model profile.
Do not rewrite missing BI fields as assumptions.
When web is used as fallback, explicitly label which facts are BI-backed, which facts are web-backed, and which facts remain unconfirmed.
