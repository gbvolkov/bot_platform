# Mycroft GAZ Subagents Service Catalog

This catalog defines the specialist subagents available to Mycroft in the GAZ-sales configuration.
Mycroft calls these subagents through the `task` delegation tool by their exact subagent names.

Use this document when a Mycroft skill needs to decide:
- which subagent can answer a request;
- what exact service to request from that subagent;
- what the subagent must not be asked to do.

## Subagent: marketing_analyst

`marketing_analyst` works with internal GAZ materials: sales presentations, sales programs, financing materials, comparison documents, configuration materials, special-body materials, TCO materials, service evidence, operations manuals, and passenger-transport materials.

Use it for meaning, positioning, arguments, applicability, program terms from materials, and document-backed sales framing.

Do not use it as the source of exact BI facts: exact price, exact payload, exact dimensions, exact trim availability, exact option availability, exact service interval, exact warranty, exact current public market data.

### marketing_analyst services

| Service | What it does | What to pass | What it returns | What it does not do |
|---|---|---|---|---|
| GAZ Family Overview | Explains which GAZ families or model directions are relevant to a broad customer task. | Customer task, cargo/passenger profile, body preference, route/work mode, constraints already known. | Candidate families/directions, why each is worth considering, internal evidence, BI/web validation flags. | Does not provide final exact model facts, prices, or availability. |
| Customer Sales Arguments | Prepares document-backed arguments for a customer, segment, or buying situation. | Model/family or task, customer type, goal, doubts or objections. | Value propositions, sales arguments, objection handling, safe customer wording. | Does not invent unsupported claims or guarantee price/availability. |
| Sales Programs And Financing Materials | Summarizes programs, promotions, subsidies, leasing, credit, FFP, and financing conditions described in internal materials. | Model/family, client type, region/country if known, quantity, purchase form, financing interest. | Program mechanics, eligibility conditions, limits, caveats, questions to confirm. | Does not guarantee current rates, approvals, actual eligibility, or bank decisions. |
| GAZ Model Sales Comparison | Compares GAZ models/families by customer fit and sales narrative. | Candidate models/families, customer task, comparison criteria. | When each option is better, strong/weak sides, positioning, caveats, facts requiring BI validation. | Does not replace BI factual comparison tables. |
| Competitor Positioning Narrative | Builds a sales narrative for GAZ versus a named competitor or competitor class using internal materials. | GAZ model/family, competitor if named, customer criteria, objection. | Internal comparison arguments, risks, weak points, claims needing BI/web validation. | Does not perform fresh market research or confirm current competitor facts. |
| Configuration Scenario Framing | Explains how body type, special body, fuel, option package, or configuration scenario should be discussed with the customer. | Task, model/family if known, body/special-body/fuel/options scenario. | Applicability logic, customer-facing explanation, caveats, BI validation flags. | Does not confirm exact option availability or technical compatibility. |
| Special Body Application Framing | Helps reason about vans, flatbeds, chassis, refrigerated bodies, workshops, buses, special vehicles, fuel variants, and other applications from materials. | Application, payload/volume constraints if known, route/road/work mode, required equipment. | Relevant directions, constraints, questions for bodybuilder or BI, internal evidence. | Does not provide engineering approval or certification. |
| Passenger Transport Fit | Explains passenger/bus directions for routes, shuttles, municipal transport, schools, corporate transport, or city/suburban service. | Route profile, passenger flow, distance, roads, stop pattern, accessibility needs, budget if known. | Passenger solution directions, route-fit reasoning, arguments, caveats, BI validation flags. | Does not confirm exact passenger capacity, regulatory compliance, or availability. |
| TCO And Operations Argumentation | Provides ownership-cost, service, operations, fuel, and reliability argumentation from internal materials. | Model/family or task, ownership period, mileage, route, fuel, service concern, competitor if relevant. | Cost drivers, service/operations arguments, risks, facts requiring BI validation. | Does not calculate exact TCO from scratch or replace BI service facts. |
| Claim Safety Check | Checks whether a proposed sales claim is supported by internal materials. | Claim text, model/family, customer context, intended use. | Safe/cautious/unsupported verdict, safer wording, evidence notes. | Does not make unsupported claims true. |
| Customer Text Drafting | Drafts short customer-facing wording based on internal materials. | Customer question, target model/family/task, tone, no-go promises. | Draft answer, email fragment, presentation wording, caveats. | Does not create a legal offer or official contract text. |

## Subagent: gaz_pricing_bi_int

`gaz_pricing_bi_int` works with the structured GAZ BI database.

Use it for formal lookup, filtering, exact prices, exact structured TTX, exact configuration/option fields, service/warranty fields, ownership-cost fields, and factual comparison of concrete candidates.

Do not use it as a sales strategist. It does not decide which option is best for a customer unless the selection criterion is formal and supplied in the request.

BI request modes:
- Analytical / comparison services return selected fields, rows, comparison columns, aggregates, and gaps needed for the task. They do not return complete model profiles and should not be asked to return all DB fields.
- Complete DB Model Profile returns the complete set of DB fields for concrete model rows, including NULL/empty fields rendered as `NA`. Mycroft forms a complete model profile from those returned rows.

Manufacturer restriction contract:
- Use `manufacturer_brand_lat` to restrict BI lookup by manufacturer.
- `manufacturer_brand_lat` must be Latin uppercase.
- For GAZ-only requests, pass `manufacturer_brand_lat=GAZ`.
- For competitor comparisons, pass a competitor manufacturer only if it is one of the database values.
- The BI database contains vehicles only for these `manufacturer_brand_lat` values: `AVIOR`, `DONGFENG`, `FOTON`, `GAZ`, `ISUZU`, `JAC`, `KAMAZ`, `LADA`, `LIAZ`, `MAZ`, `PAZ`, `PROMTEKH`, `SAZ`, `SIMAZ`, `SOLLERS`, `UAZ`, `VOLGABUS`, `YUTONG`.

### gaz_pricing_bi_int services

| Service | What it does | What to pass | What it returns | What it does not do |
|---|---|---|---|---|
| Complete DB Model Profile | Returns the complete set of DB fields for concrete model records, including NULL/empty fields. | Concrete models/modifications/candidate IDs and manufacturer scope such as `manufacturer_brand_lat=GAZ` when known. Ask BI to return every DB field/column for each matched row, preserve original DB field names where possible, include fields outside the current topic, and render every NULL/empty value as `NA`. Explicitly state that Mycroft will form a complete model profile from the returned model rows. | One complete DB-field profile per concrete model row, with values and `NA` for NULL/empty fields. | Does not choose a sales winner, perform analytics, or omit DB fields because they are null, duplicated, normalized, rare, or outside the immediate question. |
| Specific Missing Field Recovery | Re-checks BI when a field needed for the current answer was omitted from a Complete DB Model Profile or from selected analytical output. | Active concrete models/modifications/candidate IDs, manufacturer scope when known, missing field names, and likely aliases/schema names such as `front_overhang_mm`, `rear_overhang_mm`, "передний свес", "задний свес". Ask for value or `NA` if the DB value is NULL. | The requested field values if present, or `NA`/explicit absence if the DB field is NULL or unavailable, plus related original DB fields that clarify the result. | Does not replace Complete DB Model Profile when the user asks for all fields or a full model card. |
| Model Identity Lookup | Finds concrete models, modifications, or families in the database. | Model name, family name, modification, name fragment, and `manufacturer_brand_lat` when manufacturer scope is known. | `comp_full_name`, `base_model`, `comp_model`, `id`, vehicle type, body type. | Does not explain whether the model fits the customer's business need. |
| GAZ Catalog Filter | Selects GAZ records by formal parameters in analytical / comparison mode. | `manufacturer_brand_lat=GAZ`, body type, mass, payload, passenger capacity, drive, fuel, price, dimensions. Do not ask for all DB fields. | Records matching filters, selected fields needed for the filter, and relevant gaps. | Does not return complete model profiles, invent "best variants", or assign sales roles like "urban", "compromise", "best value". |
| Price Lookup | Returns prices and price ranges. | Concrete models, families, or candidate records. | `price_rub_min`, `price_rub_max`, `price_without_discounts_rub`, `price_comment`. | Does not confirm prices outside the database or interpret discounts not recorded in the database. |
| Technical Specs Lookup | Returns structured technical characteristics. | Model/candidates and required fields. | Mass, payload, dimensions, wheelbase, engine, power, gearbox, drive, fuel, wheel formula, other filled fields. | Does not infer TTX from marketing text or fill empty fields by assumption. |
| Body And Platform Lookup | Checks body versions and platform parameters. | Model/family and required body type: flatbed, van, chassis, bus, minivan. | `body_type`, platform/body parameters, cargo space dimensions, volume, pallet capacity, loading height if present. | Does not decide which body is better without formal criteria. |
| Options And Equipment Lookup | Checks option and equipment fields. | Model/candidates and list of options. | Statuses for AC, heaters, camera, parking sensors, tachograph, ERA-GLONASS, multimedia, and other fields. | Does not describe packages in advertising language or assert an option when the field is empty. |
| Service And Warranty Lookup | Returns service intervals, warranty, and service data. | Model/candidates and required service data class. | `service_interval_km`, `service_interval_months`, `warranty_months`, `warranty_km`, maintenance/parts/ownership cost if present. | Does not build a full service regulation if the database has only aggregate fields. |
| Ownership Cost Lookup | Returns available ownership-cost indicators. | Concrete models or candidates. | Ownership cost, cost per km, service components if fields are filled. | Does not calculate TCO from scratch if source inputs are absent. |
| Candidate Comparison Table | Produces a factual comparison table for concrete candidates in analytical / comparison mode. | List of models/modifications and fields to compare. Do not ask for all DB fields. | Table with selected fields such as model, price, body, mass, payload, engine, options, service data, and gaps. | Does not return complete model profiles or choose a sales winner unless the winning criterion is formal. |
| Competitor Fact Comparison | Compares GAZ and competitor models if competitor records exist in the database, in analytical / comparison mode. | Concrete GAZ models with `manufacturer_brand_lat=GAZ`, and concrete competitor models or brands with competitor `manufacturer_brand_lat` when available in the database. Do not ask for all DB fields. | Factual comparison by selected price, TTX, body, options, and other structured fields. | Does not return complete model profiles, do market analysis, or search fresh public competitor data. |
| Data Availability Check | Checks which database fields exist and which are missing. | Model/family and required attributes. | Matrix of data present / data absent by attribute. | Does not replace missing data with external facts or assumptions. |

## Subagent: web_search_agent

`web_search_agent` works with public web information.

Use it when Mycroft needs fresh/current public information, links, external verification, public competitor context, or public program data not confirmed by internal materials or BI.

Do not use it as the primary source for internal GAZ facts when BI or internal materials are the proper source.

### web_search_agent services

| Service | What it does | What to pass | What it returns | What it does not do |
|---|---|---|---|---|
| Current Public Fact Lookup | Finds current public facts from the web. | Query, entity, geography, date sensitivity, required links. | Public facts with source links and short summary. | Does not validate internal-only GAZ data. |
| Competitor Public Context | Finds current competitor model/program/positioning information. | Competitor, model, market, criteria. | Competitor facts, public source links, caveats. | Does not replace BI factual comparison when competitor exists in BI. |
| External Source Links | Collects links for claims that require public citation. | Claim or question requiring links. | Source list and supported summary. | Does not provide source-free conclusions. |
| Fresh Market Context | Checks recent market, regulatory, or public offer context. | Market question, time sensitivity, geography. | Current context and links. | Does not decide GAZ sales recommendation by itself. |

## Delegation discipline

When Mycroft calls a subagent:
- use the exact subagent name;
- pass a focused business question;
- include customer constraints already known;
- include scope such as "GAZ-only" or "competitor comparison" when relevant;
- do not pass skill names as if they were subagent names;
- do not ask BI for abstract planning, questionnaires, or sales strategy;
- do not ask marketing for exact facts that BI owns;
- do not ask web for internal-only facts.
