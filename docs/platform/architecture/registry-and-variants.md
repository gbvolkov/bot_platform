# Registry And Variants

Back to the [documentation index](../index.md).

## How the registry works

- The live registry source in this repository is `data/load.json`.
- `.env` sets `BOT_SERVICE_AGENT_CONFIG_PATH=./data/load.json`.
- `bot_service/agent_registry.py` loads this file, imports the referenced module, and expects an `initialize_agent(...)` callable.
- `/api/agents/` and `/v1/models` expose only ready agents, not every configured entry.

## Agent definition fields

Each entry in `data/load.json` can define:

- `id`
- `name`
- `description`
- `module`
- `is_active`
- `allow_raw_attachments`
- `supported_content_types`
- `streaming`
  - `modes`
  - `subgraphs`
- `params`
  - commonly `provider`, `locale`, `checkpoint_saver`, `role`, and implementation-specific keys such as `product`

## Configured IDs and implementation mapping

| Agent ID | Implementation module | Active | Notes |
|---|---|---:|---|
| `find_job` | `agents.find_job_agent` | No | Resume-to-vacancy matching flow. |
| `service_desk` | `agents.sd_ass_agent.agent` | No | Internal QA/router agent. |
| `ai_bi` | `agents.bi_agent` | No | Report and chart generation agent. |
| `new_theodor_agent` | `agents.new_theodor_agent.agent` | Yes | New product-mentor orchestrator. |
| `theodor_agent` | `agents.theodor_agent.agent` | Yes | Legacy product-mentor orchestrator. |
| `ideator` | `agents.ideator_agent` | Yes | Active report-to-ideas flow. |
| `new_ideator` | `agents.new_ideator_agent.agent` | Yes | New ideation flow with search/store tools. |
| `ideator_old` | `agents.ideator_old_agent` | No | Earlier structured ideation flow. |
| `ismart_tutor_agent` | `agents.ismart_tutor_agent` | Yes | Tutor/hint generation agent. |
| `artifact_creator_agent` | `agents.artifact_creator_agent.agent` | Yes | Artifact drafting and confirmation loop. |
| `simple_agent` | `agents.simple_agent.agent` | Yes | General-purpose prompt-configured agent. |
| `simple_agent_en` | `agents.simple_agent.agent` | Yes | English-locale variant of `simple_agent`. |
| `ismart_task_variator_agent` | `agents.ismart_task_variator_agent.agent` | Yes | Task variation generator. |
| `product_Car` | `agents.ingos_product_agent` | Yes | Product-agent variant with `product=Car`. |
| `product_Household` | `agents.ingos_product_agent` | No | Product-agent variant. |
| `product_Personal` | `agents.ingos_product_agent` | No | Product-agent variant. |
| `product_Tick Bite` | `agents.ingos_product_agent` | No | Product-agent variant. |
| `product_Инголаб` | `agents.ingos_product_agent` | No | Product-agent variant. |
| `product_Инголаб ПДФ` | `agents.ingos_product_agent` | No | Product-agent variant. |
| `product_Инголаб ППТХ` | `agents.ingos_product_agent` | No | Product-agent variant. |
| `product_Овертайм` | `agents.ingos_product_agent` | No | Product-agent variant. |
| `product_Юридическая помощь` | `agents.ingos_product_agent` | No | Product-agent variant. |

## Important distinctions

- Implementation module vs exposed model ID
  - Several IDs can map to the same implementation module with different params.
- Active vs ready
  - `is_active=true` only means the entry is eligible for startup and listing; readiness still depends on successful initialization.
- Agent folders vs registry surface
  - Support folders such as `agents/tools`, `agents/retrievers`, `agents/state`, and `agents/yandex_tools` are platform modules, not registry targets.

## Documentation rule used in this doc set

- Full implementation pages exist for each implementation module under `agents/`.
- Registry variants are consolidated here instead of duplicating nearly identical implementation pages.
