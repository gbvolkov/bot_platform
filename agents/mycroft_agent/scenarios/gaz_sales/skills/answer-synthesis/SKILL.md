---
name: answer-synthesis
description: Combine Mycroft GAZ specialist outputs into one customer-facing answer. Use after marketing, BI, web, email-preparation, or export-preparation work, and whenever source boundaries, caveats, or final recommendation consistency matter.
---

# Answer Synthesis

Use this skill when Mycroft has enough information to answer the user or must reconcile specialist outputs.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Direct tools: `../references/tools-and-actions.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- the user's current question;
- latest active customer constraints;
- prior recommendation if any;
- outputs from `marketing_analyst`, `gaz_pricing_bi_int`, and/or `web_search_agent`;
- tool results from email/export actions if relevant.

## What to do

1. Answer the user's business question first.
2. Preserve the latest active recommendation. Do not revert to an older preliminary mix unless explaining the change.
3. Separate source categories: BI-backed facts, internal marketing-material framing, public web context, assumptions, and unconfirmed gaps.
4. Resolve conflicts by source authority. BI prevails for exact BI-owned facts.
5. Filter outputs by user scope. Do not present non-GAZ models as GAZ recommendations.
6. If a final recommendation or follow-up answer needs exact price, TTX, dimensions, vehicle geometry, body/platform, cargo/loading, configuration, option, service, warranty, or ownership facts and BI has not checked the concrete candidates for the exact requested attributes, call BI before finalizing.
7. Treat BI analytical/comparison output as sufficient only for the fields it returned. Do not treat it as a complete concrete model profile.
8. If the user asks for full concrete model details, all characteristics, all options, all fields, or "строго из BI", call BI through `validate-vehicle-facts` in complete DB model profile mode before finalizing.
9. If prior BI output omitted a BI-owned field needed for the current answer, make the appropriate BI follow-up: selected-field follow-up for analytical comparison needs, or Complete DB Model Profile for concrete model-detail needs.

## What to analyze

Check:
- whether the answer covers the requested entity, attributes, detail level, and format;
- whether a candidate from marketing is missing BI confirmation where BI is required;
- whether BI output is too narrow because of synonym or body-type mismatch;
- whether web context is fresh and linked;
- whether a fleet split repeats the same model/modification without real difference;
- whether the answer would say that an exact BI-owned fact is unavailable even though BI has not been called for that exact fact in the current turn and the exact fact is not already present in visible context.
- whether visible BI context is analytical/comparison output or a complete DB model profile.
- whether the user is asking for concrete model details that require all DB fields, including NULL/empty values as `NA`.
- whether a needed BI-owned field was omitted from a previous BI result and requires the correct BI follow-up mode.

## Materials and tools

Use already collected outputs first.
Use additional `task` calls only for material gaps:
- `gaz_pricing_bi_int` for selected BI facts needed by analytical/comparison answers;
- `gaz_pricing_bi_int` through `validate-vehicle-facts` complete DB model profile mode when concrete model details or all fields are needed;
- `marketing_analyst` for missing sales framing;
- `web_search_agent` for missing current public context.

Previous BI output may be reused only for the exact fields it returned, unless it explicitly returned a Complete DB Model Profile for the active model(s). If the user asks a short or multi-attribute follow-up for exact BI-owned facts about the active model, comparison, candidate set, or fleet mix, choose the BI follow-up mode by intent: selected-field follow-up for comparison/analytics, Complete DB Model Profile for concrete model-detail retrieval.

## Output

Return one coherent user-facing answer:
- recommendation or direct answer;
- evidence and facts;
- caveats;
- next single high-impact question only if needed.

Do not expose internal coordination unless the user asks how the answer was produced.
