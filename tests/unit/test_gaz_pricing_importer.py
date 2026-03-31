import sqlite3
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "import_gaz_pricing_to_sqlite.py"
SPEC = importlib.util.spec_from_file_location("gaz_pricing_importer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
IMPORTER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER_MODULE)

SOURCE_ALLOWLIST = IMPORTER_MODULE.SOURCE_ALLOWLIST
SOURCE_DIR = IMPORTER_MODULE.SOURCE_DIR
GazPricingImporter = IMPORTER_MODULE.GazPricingImporter
build_id = IMPORTER_MODULE.build_id
canonicalize_identity_parts = IMPORTER_MODULE.canonicalize_identity_parts
normalize_id_part = IMPORTER_MODULE.normalize_id_part


def test_normalize_id_part_and_build_id():
    assert normalize_id_part("  ГАЗель NEXT  ") == "gazel_next"
    assert normalize_id_part("Соболь NN 4х4") == "sobol_nn_4h4"
    assert normalize_id_part("КАМАZ-4280-F5 (Vega)") == "kamaz_4280_f5_vega"
    assert normalize_id_part("0") is None

    assert build_id("ПАЗ 4234", "ПАЗ 4234") == "paz_4234"
    assert build_id("Газель NN", "A31R22") == "gazel_nn_a31r22"
    assert build_id(None, "   ") is None


def test_canonicalize_identity_parts_duplicates_missing_side():
    assert canonicalize_identity_parts("ПАЗ 4234", "ПАЗ 4234") == (
        "ПАЗ 4234",
        "ПАЗ 4234",
        "ПАЗ 4234",
        "paz_4234",
    )
    assert canonicalize_identity_parts("ГАЗель NN", None) == (
        "ГАЗель NN",
        "ГАЗель NN",
        "ГАЗель NN",
        "gazel_nn",
    )
    assert canonicalize_identity_parts(None, "А65R22") == (
        "А65R22",
        "А65R22",
        "А65R22",
        "a65r22",
    )


def test_merge_rows_prefers_first_non_empty_value_and_fills_gaps(tmp_path):
    importer = GazPricingImporter(tmp_path, tmp_path / "out.sqlite")
    importer.insert_normalized(
        {
            "source_file": "a.xlsx",
            "source_sheet": "БОРТ_ТТХ и состав",
            "sheet_type": "technical",
            "base_model": "Газель NEXT",
            "comp_brand": "ARGO",
            "comp_model": "SWB",
            "price_rub_min": 100.0,
            "price_rub_max": 120.0,
            "engine_fuel_type": "дизель",
        }
    )
    importer.insert_normalized(
        {
            "source_file": "b.xlsx",
            "source_sheet": "БОРТ_ТТХ и состав",
            "sheet_type": "technical",
            "base_model": "Газель NN",
            "comp_brand": "ARGO",
            "comp_model": "SWB",
            "wheelbase_mm": 3000.0,
            "engine_fuel_type": "бензин",
        }
    )

    merged_rows = importer.merge_staged_rows()

    assert len(merged_rows) == 1
    merged = merged_rows[0]
    assert merged["id"] == "argo_swb"
    assert merged["base_model"] == "ARGO"
    assert merged["comp_full_name"] == "ARGO SWB"
    assert merged["price_rub_min"] == 100.0
    assert merged["wheelbase_mm"] == 3000.0
    assert merged["engine_fuel_type"] == "дизель"
    assert merged["source_file"] == "a.xlsx"


def test_importer_warns_and_continues_when_allowlisted_sources_are_missing(tmp_path):
    db_path = tmp_path / "gaz_pricing_norm.sqlite"
    importer = GazPricingImporter(tmp_path, db_path)

    importer.run()

    assert db_path.exists()
    assert importer.stats["warnings"] == len(SOURCE_ALLOWLIST)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("select count(*) from comparisons_normalized").fetchone()[0] == 0


def test_importer_builds_single_merged_table_from_real_excels(tmp_path):
    db_path = tmp_path / "gaz_pricing_norm.sqlite"
    importer = GazPricingImporter(SOURCE_DIR, db_path)

    importer.run()

    assert importer.stats["workbooks"] == len(SOURCE_ALLOWLIST)
    assert importer.stats["selected_sheets"] == 14
    assert importer.stats["staged_rows"] == 243
    assert importer.stats["merged_rows"] == 83
    assert importer.stats["warnings"] == 0

    with sqlite3.connect(db_path) as conn:
        tables = [row[0] for row in conn.execute("select name from sqlite_master where type='table' order by name")]
        assert tables == ["comparisons_normalized"]

        row_count = conn.execute("select count(*) from comparisons_normalized").fetchone()[0]
        assert row_count == 83
        assert conn.execute("select count(*) from comparisons_normalized where id is null or trim(id) = ''").fetchone()[0] == 0
        assert conn.execute(
            "select count(*) from comparisons_normalized where ifnull(base_model, '') <> ifnull(comp_brand, '')"
        ).fetchone()[0] == 0
        assert conn.execute(
            "select count(*) from (select id, count(*) as c from comparisons_normalized group by id having c > 1)"
        ).fetchone()[0] == 0

        source_sheets = {row[0] for row in conn.execute("select distinct source_sheet from comparisons_normalized")}
        expected_sheets = {
            "ПАЗ (расширенные)",
            "БОРТ_ТТХ и состав",
            "ЦМФ_ТТХ и состав",
            "АВТОБУС_ТТХ и состав",
            "LDT_ТТХ и состав",
            "Минивен",
        }
        assert source_sheets == expected_sheets

        assert conn.execute("select count(*) from comparisons_normalized where id = 'citymax_8'").fetchone()[0] == 1
        assert conn.execute("select count(*) from comparisons_normalized where id = 'argo_swb'").fetchone()[0] == 1
        assert conn.execute("select count(*) from comparisons_normalized where id = 'donfeng_c80'").fetchone()[0] == 1


def test_downstream_defaults_point_to_normalized_db_and_single_table_contract():
    repo_root = Path(__file__).resolve().parents[2]
    run_bi_text = (repo_root / "run_bi_agent_cli.py").read_text(encoding="utf-8")
    agent_text = (repo_root / "agents" / "gaz_agent" / "agent.py").read_text(encoding="utf-8")
    prompt_text = (repo_root / "agents" / "gaz_agent" / "pricing_bi_prompt_context.txt").read_text(encoding="utf-8")

    assert "gaz_pricing_norm.sqlite" in run_bi_text
    assert "gaz_pricing_norm.sqlite" in agent_text
    assert "TABLE: comparison_model_options" not in prompt_text
    assert "TABLE: comparison_service_groups" not in prompt_text
    assert "TABLE: comparisons_raw_params" not in prompt_text
    assert "comp_model" in prompt_text
    assert "comp_brand` mirrors `base_model`" in prompt_text
    assert "TABLE: comparisons_normalized" in prompt_text
