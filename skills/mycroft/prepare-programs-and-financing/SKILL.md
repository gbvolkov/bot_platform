---
name: prepare-programs-and-financing
description: Explain GAZ sales programs, promotions, subsidies, FFP, leasing, credit, financing conditions, or discount/offer terms from internal materials and related sources. Use when the user asks how a deal can be financed or which program may apply.
---

# Prepare Programs And Financing

Use this skill for sales-program and financing questions.

Reference documents:
- Subagent services: `../references/subagents-service-catalog.md`
- Full skill catalog: `../references/skill-catalog.md`

## Input

Receive:
- model/family if known;
- customer type;
- region/country if known;
- quantity;
- purchase form;
- financing preference;
- budget, term, advance payment, or timing if provided.

## What to do

1. Ask `marketing_analyst` for internal program and financing material summary.
2. Ask BI for vehicle prices if the financing answer depends on a vehicle price base.
3. Ask web only for current public program/rate context or source links.
4. Do not guarantee eligibility, approval, availability, or current bank rates unless a source confirms them.

## What to analyze

Check:
- whether the program is from internal materials or current public sources;
- whether time sensitivity requires web validation;
- whether exact vehicle price is needed;
- which terms must be confirmed by dealer, bank, or program owner.

## Materials and tools

Use `task`:
- `marketing_analyst`: Sales Programs And Financing Materials, Customer Text Drafting, Claim Safety Check.
- `gaz_pricing_bi_int`: Price Lookup, Model Identity Lookup, Data Availability Check.
- `web_search_agent`: Current Public Fact Lookup, External Source Links.

## Output

Return:
- possible program/financing paths;
- conditions and limitations;
- required confirmations;
- customer-facing wording if requested.
