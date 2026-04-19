---
name: landscape-workflow
description: Build a broad product-direction or segment landscape from internal GAZ marketing materials; call marketing document tools before answering.
---

# Landscape Workflow

Apply this workflow for broad internal-materials discovery.

Do not answer a landscape, shortlist, model-selection, or segment-fit request from memory. Before the final answer, use at least one configured marketing document tool such as `search_marketing_materials`, `get_marketing_branch_pack`, or `read_marketing_material`. If no relevant material is found, say that internal marketing evidence is unavailable or weak.

Preferred steps:
1. Use `estimate_marketing_research_cost` if the request is broad or underspecified.
2. Use `search_marketing_materials` with a clear intent such as `landscape`, `positioning`, `use_case`, or `recommendation`.
3. Inspect returned candidate metadata before choosing directions: `product_families`, `doc_kind`, `transport_type`, `branches`, `body_tags`, and `competitor_tags`.
4. If the customer did not name a family, start broad. If results are too narrow or unrelated, rerun with better natural-language terms and, when justified, family filters.
5. If the request has a clear body or segment, also use `get_marketing_branch_pack` with slots such as `transport_type`, `body_type`, `capacity_or_payload`, `route_type`, and `customer_goal`.
6. Use `read_marketing_material` only for the most relevant candidates.
7. Return a supported candidate set when the task is about model/family fit: candidate model/family, why it fits, supporting material candidate, caveats, and exact facts that require BI validation.
8. Summarize fit, use cases, sales techniques, objections, trade-offs, discount/offer conditions when present, and evidence strength.

Query formulation:
- For cargo delivery and open-board tasks, include both business words and index terms, for example `городская доставка`, `борт`, `бортовая платформа`, `платформа`, `тент`, `flatbed`, `cargo`, `грузоподъемность`, `маневренность`.
- Use canonical family filters only when the family is named or already surfaced by evidence, for example `gazelle_next`, `gazelle_nn`, `gazelle_business`, `sobol_nn`, `gazon_next`, `valdai`, `sadko`, `vector_next`, `gazelle_city`, `paz`.
- If obvious families are missing from results, do not fill the gap from memory. Say that indexed marketing coverage looks incomplete or weak and return the best supported candidates.

For fleet-selection and model-fit tasks, do not finish with only a research plan, parameter checklist, or generic request template. If candidate evidence is weak, return the best supported candidates plus evidence gaps; if no candidates are supported, say that explicitly.

Return compact findings for the manager. Do not write a final sales pitch unless explicitly asked.
