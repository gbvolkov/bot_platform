from __future__ import annotations

from pathlib import Path
import json

from deepagents.middleware.skills import _list_skills

from agents.mycroft_agent.cli_config import (
    InternalToolSpec,
    SkillsConfig,
    SubagentsConfig,
    build_internal_tools,
    load_cli_config,
)
from agents.mycroft_agent.configured_agent import build_skills_backend, normalize_skill_source


CONFIG_PATH = Path("agents/mycroft_agent/scenarios/kpi_agent/config.json")
SKILLS_PATH = "agents/mycroft_agent/scenarios/kpi_agent/skills"
PROMPT_PATH = Path("agents/mycroft_agent/scenarios/kpi_agent/system_prompt.txt")
KPI_AGENT_CATALOG_PATH = Path(
    "agents/mycroft_agent/scenarios/kpi_agent/references/kpi-agent-service-catalog.md"
)
SUBAGENT_CATALOG_PATH = Path(
    "agents/mycroft_agent/scenarios/kpi_agent/references/subagents-service-catalog.md"
)
KPI_BI_PROMPT_PATH = Path("agents/kpi_agent/kpi_bi_prompt_context.txt")
KPI_DB_DESCRIPTION_PATH = Path("agents/kpi_agent/kpi_db_description.md")


def test_kpi_agent_config_loads_kpi_bi_stateless_subagent_only():
    config = load_cli_config(CONFIG_PATH)

    assert config.subagents == SubagentsConfig(
        stateless=("kpi_bi_int",),
        stateful=(),
    )
    assert config.skills == SkillsConfig(paths=(SKILLS_PATH,))
    assert config.internal_tools == (
        InternalToolSpec(
            import_path=(
                "agents.mycroft_agent.scenarios.kpi_agent.tools:"
                "build_kpi_staff_structure_fuzzy_search_tool"
            ),
            params={},
        ),
    )
    assert config.mcp.servers == ()
    assert config.deepagents.interrupt_on == {}


def test_kpi_agent_is_registered_as_public_configured_mycroft_agent():
    load_config = json.loads(
        Path("data/config/bot_service/load.json").read_text(encoding="utf-8")
    )
    kpi_agent = next(agent for agent in load_config["agents"] if agent["id"] == "kpi_agent")
    kpi_bi = next(agent for agent in load_config["agents"] if agent["id"] == "kpi_bi_int")

    assert kpi_agent["module"] == "agents.mycroft_agent.configured_agent"
    assert kpi_agent["is_active"] is True
    assert kpi_agent["supported_content_types"] == []
    assert kpi_agent["params"]["provider"] == "openai"
    assert kpi_agent["params"]["config_path"] == str(CONFIG_PATH).replace("\\", "/")
    assert kpi_agent["params"]["model_size"] == "base"
    assert kpi_agent["params"]["temperature"] == 0.1
    assert kpi_agent["params"]["streaming"] is False
    assert kpi_agent["params"]["checkpoint_saver"] == "SQLite"
    assert kpi_bi["is_active"] is False


def test_kpi_agent_prompt_defines_position_first_database_boundaries():
    config = load_cli_config(CONFIG_PATH)
    prompt = config.system_prompt.lower()

    assert "position-first" in prompt
    assert "kpi_staff_structure_fuzzy_search" in prompt
    assert "kpi_bi_int" in prompt
    assert "`kpi_bi_int` is stateless" in prompt
    assert "kpi_values" in prompt
    assert "kpi_staff_structure" in prompt
    assert "kpi_method_ref" in prompt
    assert "kpi metrics and formula terms" in prompt
    assert "when inferred from candidates, the context must be confirmed by the user" in prompt
    assert "every kpi-list answer must state the official applied position name" in prompt
    assert "if bi returns n assignment rows" in prompt
    assert "exactly\n  n numbered kpi assignment items" in prompt
    assert "never merge, deduplicate, group, aggregate" in prompt
    assert "one numbered item per bi assignment row" in prompt
    assert "\"compact\" means shorter wording inside each assignment item" in prompt
    assert "must call `task` with `subagent_type: kpi_bi_int`" in prompt
    assert "never call `task` with `subagent_type: general-purpose`" in prompt
    assert "staff-structure-first rule" in prompt
    assert "personal kpi question" in prompt
    assert "without any usable staff-structure clue" in prompt
    assert "do not call fuzzy search" in prompt
    assert "do not call\n   bi" in prompt
    assert "pronouns and generic words are not usable clues" in prompt
    assert "`results[0].staff_structure_ids` are matching row identifiers" in prompt
    assert "make a staff structure validation" in prompt
    assert "for every returned `staff_structure_id`" in prompt
    assert "do not use fuzzy-only fields as final user-facing options" in prompt
    assert "do not query `kpi_values` during staff-structure validation" in prompt
    assert "every `task` request to `kpi_bi_int` must be self-contained" in prompt
    assert "never ask bi about \"item 1\", \"item 10\", \"item 13\", \"the" in prompt
    assert "use `task` with `subagent_type: kpi_bi_int` only for concrete data requests" in prompt
    assert "never ask `kpi_bi_int` to decide what mycroft should ask the user" in prompt
    assert "do not ask bi to search `kpi_values` until the exact staff-structure context is resolved" in prompt
    assert "do not mention bi calls, fuzzy search, tools, subagents, sql, or internal" in prompt
    assert "in user-facing conversation, never mention databases, database files, source" in prompt
    assert "refer only to the corporate kpi methodology\n  and the official corporate" in prompt
    assert "this privacy rule applies only to conversation with the user" in prompt
    assert "in internal\n  requests to bi, subagents, or tools, use exact technical names" in prompt
    assert "do not ask the user for permission to use internal tools or subagents" in prompt
    assert "employee group means a group of staff" in prompt
    assert "position means the job or role" in prompt
    assert "`position_group` means position level" in prompt
    assert "do not expose database field names" in prompt
    assert "translate technical fields into business wording" in prompt
    assert "primary" in prompt
    assert "secondary" in prompt
    assert "no official split" in prompt


def test_kpi_agent_catalogs_describe_scenarios_and_subagent_services():
    agent_catalog = KPI_AGENT_CATALOG_PATH.read_text(encoding="utf-8").lower()
    subagent_catalog = SUBAGENT_CATALOG_PATH.read_text(encoding="utf-8").lower()

    assert "identify position context" in agent_catalog
    assert "explain kpis for position" in agent_catalog
    assert "explain kpi metrics" in agent_catalog
    assert "navigate org structure" in agent_catalog
    assert "mycroft must not call tools for a personal kpi question" in agent_catalog
    assert "full structure for each candidate" in agent_catalog
    assert "position lookup" in subagent_catalog
    assert "staff structure validation" in subagent_catalog
    assert "one official `kpi_staff_structure` row per candidate id" in subagent_catalog
    assert "kpi assignment lookup" in subagent_catalog
    assert "kpi method lookup" in subagent_catalog
    assert "kpi metric text search" in subagent_catalog
    assert "data availability check" in subagent_catalog
    assert "`position_group` means position level" in agent_catalog
    assert "`position_group` is the position level" in subagent_catalog
    assert "what should i ask the user" in subagent_catalog
    assert "list possible positions for the user" in subagent_catalog


def test_kpi_position_skills_require_confirmed_position_before_kpi_list():
    position_skill = Path(
        "agents/mycroft_agent/scenarios/kpi_agent/skills/"
        "kpi-position-identification/SKILL.md"
    ).read_text(encoding="utf-8").lower()
    list_skill = Path(
        "agents/mycroft_agent/scenarios/kpi_agent/skills/"
        "kpi-list-assigned-kpis/SKILL.md"
    ).read_text(encoding="utf-8").lower()
    synthesis_skill = Path(
        "agents/mycroft_agent/scenarios/kpi_agent/skills/"
        "kpi-answer-synthesis/SKILL.md"
    ).read_text(encoding="utf-8").lower()

    assert "no usable clue in personal kpi request" in position_skill
    assert "do not call fuzzy search. do not call bi" in position_skill
    assert "until the exact official staff-structure context is resolved" in position_skill
    assert "do not request or\n   return kpi assignments" in list_skill
    assert "call `task` with `subagent_type: kpi_bi_int`" in list_skill
    assert "if `kpi_bi_int` has not returned assignment rows" in list_skill
    assert "ask bi before" in list_skill
    assert "always state the official position name" in list_skill
    assert "one numbered item per assignment row returned by bi" in list_skill
    assert "not one item per\n  unique kpi name" in list_skill
    assert "numbered list must contain exactly n" in list_skill
    assert "never merge, deduplicate, group, aggregate" in list_skill
    assert "compact wording is allowed only inside each assignment item" in list_skill
    assert "prefer a validated `staff_structure_id` as the kpi lookup scope" in list_skill
    assert "every kpi-list answer must state the official applied position name" in synthesis_skill
    assert "preserve the bi assignment row count" in synthesis_skill
    assert "must contain exactly n kpi" in synthesis_skill
    assert "do not merge, deduplicate, group, aggregate" in synthesis_skill
    assert "repeated kpi names with different calculation detail" in synthesis_skill
    assert "compare the number of numbered kpi items" in synthesis_skill
    assert "employee group means a group of staff within a department" in position_skill
    assert "`position_group` means position level" in position_skill
    assert "must be confirmed by `kpi_bi_int`" in position_skill
    assert "these bi calls must ask about `kpi_staff_structure`, not `kpi_values`" in position_skill
    assert "do not request kpi assignment lookup and do not ask for `kpi_values` rows" in position_skill
    assert "`staff_structure_ids` are matching row identifiers in the same order" in position_skill
    assert "before asking the user about candidates" in position_skill
    assert "for all candidate ids returned by fuzzy search" in position_skill
    assert "user-facing options must come from bi-validated rows" in position_skill
    assert "return complete hierarchy for these candidate ids" in position_skill
    assert "validate these `staff_structure_id` values and return exact rows" in position_skill
    assert "do not ask the user for permission to use fuzzy search, bi, tools, or\n  subagents" in position_skill
    assert "do not ask bi what mycroft should ask the user" in position_skill


def test_kpi_skills_make_bi_subagent_mandatory_for_database_facts():
    scenario_root = Path("agents/mycroft_agent/scenarios/kpi_agent")
    source_policy = (
        scenario_root / "skills/kpi-source-policy/SKILL.md"
    ).read_text(encoding="utf-8").lower()
    method_skill = (
        scenario_root / "skills/kpi-method-explanation/SKILL.md"
    ).read_text(encoding="utf-8").lower()
    structure_skill = (
        scenario_root / "skills/kpi-structure-navigation/SKILL.md"
    ).read_text(encoding="utf-8").lower()

    assert "subagent_type: kpi_bi_int` is mandatory" in source_policy
    assert "mycroft owns intent classification" in source_policy
    assert "if a personal kpi request has no usable clue" in source_policy
    assert "never call `task` with `subagent_type: general-purpose`" in source_policy
    assert "bi calls before kpi lookup must ask only" in source_policy
    assert "enrich and validate every candidate by id before user dialogue" in source_policy
    assert "use bi-validated staff-structure rows" in source_policy
    assert "forbidden bi requests" in source_policy
    assert "do not ask permission before using internal tools or subagents" in source_policy
    assert "call `task` with `subagent_type: kpi_bi_int`" in method_skill
    assert "call `task` with `subagent_type: kpi_bi_int`" in structure_skill
    assert "`kpi_bi_int` is stateless" in method_skill
    assert "do not ask bi about \"item 1\", \"point 10\"" in method_skill
    assert "make the bi request self-contained" in method_skill
    assert "never assume that bi remembers the previous kpi list or previous numbering" in method_skill
    assert "staff_structure_id=8" in method_skill
    assert "те же kpi name + business line +" in method_skill
    assert "pool flag + calculation detail + specifics" in method_skill
    assert "complete staff structure for each candidate id" in structure_skill
    assert "do not ask the user to choose from fuzzy-only fields" in structure_skill


def test_kpi_answer_skills_hide_database_field_names_from_users():
    synthesis_skill = Path(
        "agents/mycroft_agent/scenarios/kpi_agent/skills/"
        "kpi-answer-synthesis/SKILL.md"
    ).read_text(encoding="utf-8").lower()
    structure_skill = Path(
        "agents/mycroft_agent/scenarios/kpi_agent/skills/"
        "kpi-structure-navigation/SKILL.md"
    ).read_text(encoding="utf-8").lower()

    assert "do not expose database field names" in synthesis_skill
    assert "convert technical names to human labels" in synthesis_skill
    assert "`position_group` -> \"уровень должности\"" in synthesis_skill
    assert "do not show raw database field names" in structure_skill
    assert "do not expose tool names, bi calls, fuzzy-search calls, subagent calls" in synthesis_skill
    assert "do not ask the user for permission to use internal tools or subagents" in synthesis_skill
    assert "second-level department" in structure_skill


def test_kpi_agent_defines_position_group_as_position_level():
    scenario_root = Path("agents/mycroft_agent/scenarios/kpi_agent")
    checked_paths = [
        scenario_root / "system_prompt.txt",
        scenario_root / "references/kpi-agent-service-catalog.md",
        scenario_root / "references/subagents-service-catalog.md",
        scenario_root / "skills/kpi-source-policy/SKILL.md",
        scenario_root / "skills/kpi-position-identification/SKILL.md",
        scenario_root / "skills/kpi-list-assigned-kpis/SKILL.md",
        scenario_root / "skills/kpi-structure-navigation/SKILL.md",
        scenario_root / "skills/kpi-answer-synthesis/SKILL.md",
    ]
    lines = [
        line.lower()
        for path in checked_paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    combined = "\n".join(lines)

    assert not any("role group" in line for line in lines)
    assert "not a field" not in combined
    assert "does not have a `position_group` field" not in combined
    assert "position_group` means position level" in combined
    assert "position_group` -> \"уровень должности\"" in combined


def test_kpi_bi_prompt_defines_position_group_as_position_level():
    bi_prompt = KPI_BI_PROMPT_PATH.read_text(encoding="utf-8").lower()
    db_description = KPI_DB_DESCRIPTION_PATH.read_text(encoding="utf-8").lower()

    assert "`position_group` means position level for the kpi assignment" in bi_prompt
    assert "position levels such as" in bi_prompt
    assert "for every string search predicate" in bi_prompt
    assert "normalize both sides in sql" in bi_prompt
    assert "lower(trim(field_name" in bi_prompt
    assert "this applies to `like`, equality, and `in` comparisons" in bi_prompt
    assert "`staff_structure_id` is the primary key of `kpi_staff_structure`" in bi_prompt
    assert "one\n  `staff_structure_id` identifies exactly one official staff-structure row" in bi_prompt
    assert "if the user request already contains a concrete `staff_structure_id`" in bi_prompt
    assert "do not add extra text filters" in bi_prompt
    assert "`kpi_values.staff_structure_ref = <staff_structure_id>`" in bi_prompt
    assert "where v.staff_structure_ref = 16" in bi_prompt
    assert "group of positions" not in bi_prompt
    assert "role groups" not in bi_prompt
    assert "`position_group` означает уровень должности" in db_description
    assert "группа должностей" not in db_description


def test_kpi_agent_skills_backend_lists_scenario_skills():
    virtual_skills_path = normalize_skill_source(SKILLS_PATH)
    backend = build_skills_backend((virtual_skills_path,))

    skills = _list_skills(backend, virtual_skills_path)

    assert {skill["name"] for skill in skills} == {
        "kpi-answer-synthesis",
        "kpi-list-assigned-kpis",
        "kpi-method-explanation",
        "kpi-position-identification",
        "kpi-source-policy",
        "kpi-structure-navigation",
    }


def test_kpi_staff_structure_fuzzy_search_tool_returns_exact_candidates():
    config = load_cli_config(CONFIG_PATH)
    tools = build_internal_tools(config.internal_tools)

    tool = next(item for item in tools if item.name == "kpi_staff_structure_fuzzy_search")
    raw_result = tool.invoke(
        {
            "query": "\u043e\u0442\u0434\u0435\u043b \u0440\u0430\u0431\u043e\u0442\u044b \u0441 \u0431\u0430\u043d\u043a\u0430\u043c\u0438",
            "limit_per_field": 5,
        }
    )
    result = json.loads(raw_result)

    assert result["query"] == "\u043e\u0442\u0434\u0435\u043b \u0440\u0430\u0431\u043e\u0442\u044b \u0441 \u0431\u0430\u043d\u043a\u0430\u043c\u0438"
    assert result["search_mode"] == "full_position_name"
    assert result["results"] == [
        {
            "db_fieldname": "full_position_name",
            "candidate_values": [
                candidate["full_position_name"] for candidate in result["candidates"]
            ],
            "staff_structure_ids": [
                candidate["staff_structure_id"] for candidate in result["candidates"]
            ],
        }
    ]
    assert any(
        "\u041e\u0442\u0434\u0435\u043b \u043f\u043e \u0440\u0430\u0431\u043e\u0442\u0435 \u0441 \u0431\u0430\u043d\u043a\u0430\u043c\u0438"
        in candidate["full_position_name"]
        and candidate["fields"]["department_2"]
        == "\u041e\u0442\u0434\u0435\u043b \u043f\u043e \u0440\u0430\u0431\u043e\u0442\u0435 \u0441 \u0431\u0430\u043d\u043a\u0430\u043c\u0438"
        for candidate in result["candidates"]
    )


def test_kpi_staff_structure_fuzzy_search_tool_matches_full_position_names():
    config = load_cli_config(CONFIG_PATH)
    tools = build_internal_tools(config.internal_tools)

    tool = next(item for item in tools if item.name == "kpi_staff_structure_fuzzy_search")
    raw_result = tool.invoke(
        {
            "query": "\u0443\u0440\u0435\u0433\u0443\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0443\u0431\u044b\u0442\u043a\u043e\u0432",
            "limit_per_field": 8,
        }
    )
    result = json.loads(raw_result)

    assert result["search_mode"] == "full_position_name"
    assert len(result["candidates"]) > 1
    assert all("full_position_name" in candidate for candidate in result["candidates"])
    assert result["results"][0]["staff_structure_ids"] == [
        candidate["staff_structure_id"] for candidate in result["candidates"]
    ]
    assert 90 in result["results"][0]["staff_structure_ids"]
    assert any(
        candidate["fields"].get("department_2")
        == "\u041e\u0442\u0434\u0435\u043b \u0443\u0440\u0435\u0433\u0443\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f \u0443\u0431\u044b\u0442\u043a\u043e\u0432"
        for candidate in result["candidates"]
    )
    assert any(
        candidate["staff_structure_id"] == 90
        and "\u041e\u0421\u0410\u0413\u041e, \u0418\u0424\u041b, \u0412\u0417\u0420"
        in candidate["full_position_name"]
        for candidate in result["candidates"]
    )
