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
7. Choose the BI request mode before calling BI.
8. Use **selected fact mode** when the user asks for one or several exact attributes, a factual comparison table, catalog filtering, ownership/service analytics, or "which is better/cheaper" by known criteria. In this mode, request only the needed fields or comparison columns. State in the BI prompt that this is not a complete model profile request and BI does not need to return all DB fields.
9. Use **complete DB model profile mode** when the user asks for a concrete model's full details, full characteristics, options, complete BI list, "все поля", "полный список", "строго из BI", or when a short follow-up needs concrete model details that were absent from an earlier analytical result.
10. In complete DB model profile mode, require BI to return the complete set of DB fields for every matched concrete model row, including fields with NULL/empty values. Ask BI to render every NULL/empty value as `NA`, preserve original DB field names where possible, and include fields even when they are outside the current topic.
11. Explicitly state in the BI request that Mycroft will form a complete model profile from the returned model rows, so BI must not drop columns merely because they are null, normalized, rarely used, or not needed for the immediate answer.
12. Do not infer BI absence from prior BI output that was selected fact mode, comparison mode, or otherwise did not return the exact field. Before saying "нет в BI", "не возвращается", or "BI does not contain this", make the correct BI request mode for the active model(s).
13. If a Complete DB Model Profile was requested but the returned field list still omits a needed field, immediately make a targeted BI follow-up for the active model(s) and the specific missing DB/user-facing field(s). Include likely aliases or schema names when known, for example for overhangs ask for `front_overhang_mm`, `rear_overhang_mm`, "передний свес", and "задний свес"; ask BI to return the value or `NA` if the DB value is NULL.
14. If BI returns no candidates or no records for marketing candidates, do not stop there. Make one corrected BI follow-up request using more formal lookup terms: concrete family names, model name fragments, manufacturer scope through `manufacturer_brand_lat`, body/platform type, payload or mass range, drive, fuel, price range, or other structured filters already known from the user or marketing evidence.
15. When reformulating after an empty BI result, explicitly ask BI to check whether the issue is naming, manufacturer restriction, body-type synonym, missing field coverage, or true absence of matching records.
16. If the corrected BI follow-up still lacks a required fact, decide whether the missing fact is public/current and can be checked externally. If yes, use `web_search_agent` only for that missing fact or source links.
17. Do not use web to override a BI value. Use web only to fill BI gaps, provide public links, or check current public facts.
18. Do not ask BI to choose a sales winner without formal criteria.

## Routing shortcut

If the user names a competitor manufacturer that is present in BI, use BI before web for structured vehicle facts and factual comparison.
For example, for Dongfeng use `manufacturer_brand_lat=DONGFENG` and ask BI for candidate identity, body/platform, GVW, wheelbase, payload, options, service fields, and gaps before using web for fresh public links or fields missing from BI.

## What to analyze

Check:
- whether BI returned the requested exact fields;
- whether the BI request mode was selected fact mode or complete DB model profile mode;
- whether selected fact mode avoided asking for all DB fields;
- whether complete DB model profile mode returned every DB field, including NULL/empty fields as `NA`;
- whether records are in scope;
- whether manufacturer scope was passed through `manufacturer_brand_lat` when relevant;
- whether missing fields affect the answer;
- whether a previous BI result was only analytical/comparison output, not a complete model profile;
- whether the current answer needs a field omitted from a Complete DB Model Profile, requiring a targeted BI re-query for that exact field and aliases;
- whether a marketing candidate was missed due to naming/body synonym;
- whether a follow-up BI request is needed;
- whether an empty BI result has been retried with concrete families, body/platform type, and formal filters before treating the fact as unavailable;
- whether any remaining missing fact should be checked through web because it is public, current, or requires source links.

## Materials and tools

Use `task -> gaz_pricing_bi_int` first.
Use `task -> web_search_agent` only after BI and corrected BI follow-up are incomplete, and only for missing public/current facts or source links.

Useful services:
- Complete DB Model Profile;
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

Return BI-backed facts for selected fact mode, or a complete model profile formed from all returned DB fields in complete DB model profile mode.
Do not rewrite missing BI fields as assumptions.
When web is used as fallback, explicitly label which facts are BI-backed, which facts are web-backed, and which facts remain unconfirmed.
