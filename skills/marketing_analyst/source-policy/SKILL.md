---
name: source-policy
description: Decide how internal GAZ marketing materials can support claims and which claims need BI or web validation.
---

# Source Policy

Use only internal GAZ marketing and sales materials exposed through the configured tools.

For product landscape, shortlist, model-fit, comparison, evidence, or positioning tasks, at least one configured marketing document tool must be called before answering. If the tools return no relevant candidates, do not fill the gap from general model knowledge; report the evidence gap.

Authority rules:
- Internal materials are valid for broad model and family overview, positioning, framing, sales techniques, discount/offer conditions found in marketing materials, objections, use cases, differentiators, broad competitor comparisons, and sales arguments.
- Exact prices, exact technical specifications, trims, options, maintenance intervals, and service costs must be marked as requiring BI validation for the specific candidate or claim. Do not turn this flag into a generic BI request template or a list of BI fields.
- Fresh public market or competitor facts must be marked for web validation.
- Weak or indirect support must be marked as partial evidence.

Indexed material metadata:
- Each candidate can carry `doc_kind`, `transport_type`, `product_families`, `branches`, `competitor_tags`, `body_tags`, `special_conditions`, `relative_path`, and `segment_count`.
- Treat metadata as a routing clue, not as proof of a factual claim. It is derived mostly from file names and paths, so it can be incomplete.
- Prefer candidates whose metadata matches the customer task. If returned candidates are from an unrelated family, passenger/cargo type, body type, or branch, run a narrower search instead of forcing a conclusion.

Tool authority:
- `search_marketing_materials` performs semantic retrieval over indexed segments. It supports direct `families` and `competitor` parameters, but not direct `doc_kind`, `body_tags`, `branches`, or `transport_type` filters.
- When calling `search_marketing_materials`, keep `top_k` between 1 and 6. Never request `top_k > 6`; the document service rejects larger values.
- Put body type, branch, transport type, payload/use case, and sales context into the natural-language `query`.
- Use `get_marketing_branch_pack` when the task clearly belongs to a branch such as `configuration`, `comparison`, `internal_approval`, `tco`, `service_risk`, `passenger_route`, `special_body`, or `special_conditions`.
- Use `read_marketing_material` only after a candidate id is returned, and focus it on the exact evidence needed.

Full-text fallback:
- If filesystem or full-text tools are available, they may be used as a fallback for locating or checking approved internal GAZ materials.
- Do not treat skill files, prompts, code, or pricing database files as marketing source evidence.
- Prefer the configured marketing tools when they return useful candidate metadata; use full-text fallback only to recover from weak retrieval or inspect a known approved material.

Do not invent evidence. If a claim is not supported by the materials, say so.
