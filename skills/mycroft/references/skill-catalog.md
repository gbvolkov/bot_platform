# Mycroft GAZ Skill Catalog

This catalog lists the Mycroft orchestration skills used in the GAZ-sales configuration.

Each skill must keep its `SKILL.md` body focused on the workflow and refer to:
- `../references/subagents-service-catalog.md` for subagent services;
- `../references/tools-and-actions.md` for direct tools.

## Skills

| Skill | Use when | Main outputs |
|---|---|---|
| `source-authority-and-routing` | Mycroft must decide which source or subagent owns the answer. | Source plan and delegation order. |
| `capture-customer-requirements` | The customer describes a vehicle/fleet/business task and the useful inputs must be extracted. | Known inputs, assumptions, missing high-impact inputs. |
| `shortlist-gaz-solutions` | The customer needs GAZ directions, families, or candidate models before exact BI validation. | Candidate directions and validation needs. |
| `validate-vehicle-facts` | Exact selected facts are needed, or the user asks for a complete concrete model profile/all DB fields. | BI-backed selected facts, or a complete model profile formed from all returned DB fields with `NA` for nulls. |
| `build-vehicle-recommendation` | Mycroft must recommend a model, configuration, or fleet mix. | Recommendation with source boundaries and caveats. |
| `compare-customer-options` | The user asks to compare GAZ models, configurations, body types, fuels, or purchase options. | Analytical comparison using selected BI fields, not complete model profiles. |
| `prepare-sales-argumentation` | The user needs customer-facing sales arguments, objection handling, or safe wording. | Arguments, caveats, safe claims. |
| `handle-competitor-comparison` | The user asks to compare GAZ with a competitor or respond to competitor pressure. | Internal narrative, BI facts, web facts if needed. |
| `prepare-programs-and-financing` | The user asks about promotions, sales programs, subsidies, FFP, leasing, credit, or financing terms. | Program/financing summary and confirmation needs. |
| `build-tco-case` | The user asks about ownership cost, operations, service cost, fuel, or economic justification. | TCO/operations case with confirmed facts and assumptions. |
| `answer-service-or-operation-question` | The user asks about service, warranty, maintenance, operation, restrictions, or reliability concerns. | Service/operation answer with exact facts and caveats. |
| `answer-synthesis` | Mycroft has specialist outputs and must create one final answer. | Final customer-facing answer. |
| `email-followup` | The user asks to draft or send a follow-up email. | Draft or sent email through Gmail tools. |
| `artifact-export` | The user asks to save/export/persist a result. | Stored artifact link. |

## General interaction rule

Use the narrowest skill that matches the user's current turn.
If several skills are involved, use them in this sequence:
1. capture requirements;
2. route to sources;
3. collect marketing/BI/web evidence;
4. synthesize the answer;
5. perform email/export action only if explicitly requested.
