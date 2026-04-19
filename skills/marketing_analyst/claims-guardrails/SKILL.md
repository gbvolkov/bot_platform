---
name: claims-guardrails
description: Classify marketing claims by evidence strength and validation requirements.
---

# Claims Guardrails

Classify important claims before returning them to the manager.

Evidence strength:
- High: directly supported by multiple strong internal excerpts.
- Medium: supported by one relevant excerpt or indirectly supported by materials.
- Low: weakly implied, partial, or ambiguous.

Validation flags:
- `requires_bi_validation`: exact price, exact specification, trim, option, maintenance interval, service cost, or numeric fact from BI-owned data.
- `requires_web_validation`: fresh public market fact, external competitor claim, current availability, or external source claim.

Do not upgrade evidence strength to high without direct internal support.
