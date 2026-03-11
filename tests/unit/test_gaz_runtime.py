from pathlib import Path
from types import SimpleNamespace
import shutil
import uuid

import pytest

from services.kb_manager.gaz_runtime import GazRuntimeService


@pytest.fixture
def gaz_runtime_fixture(monkeypatch):
    base_dir = Path("logs") / f"gaz-runtime-{uuid.uuid4().hex}"
    docs_root = base_dir / "docs"
    cache_root = base_dir / "cache"
    docs_root.mkdir(parents=True)

    files = {
        "gazelle_nn_base_and_options.pdf": "Gazelle NN engine 2.5 turbo diesel 149 hp. Payload baseline and body options.",
        "sobol_nn_operations_manual.pdf": "Sobol NN engine family with fuel consumption notes and maintenance intervals.",
        "product_landscape.pdf": "Urban delivery portfolio overview with financing support and service coverage.",
    }

    for name in files:
        (docs_root / name).write_text("stub", encoding="utf-8")

    content_by_path = {
        str((docs_root / name).resolve()): [SimpleNamespace(page_content=text)]
        for name, text in files.items()
    }

    def fake_loader(file_path: str):
        return content_by_path[str(Path(file_path).resolve())]

    monkeypatch.setattr("services.kb_manager.gaz_runtime.load_single_document", fake_loader)
    monkeypatch.setattr(GazRuntimeService, "_build_rag_index", lambda self, collection_id, manifest, material_artifacts: True)

    service = GazRuntimeService(docs_root=docs_root, cache_root=cache_root)
    rebuild = service.rebuild_collection("gaz")
    status = service.collection_status("gaz")
    assert rebuild["material_artifacts_built"] is True
    assert status["material_artifacts_built"] is True

    def fail_loader(_file_path: str):
        raise AssertionError("runtime attempted raw document read")

    monkeypatch.setattr("services.kb_manager.gaz_runtime.load_single_document", fail_loader)
    manifest = service._load_manifest("gaz")
    try:
        yield service, manifest
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_read_material_uses_cached_artifacts(gaz_runtime_fixture):
    service, manifest = gaz_runtime_fixture
    candidate_id = next(item["candidate_id"] for item in manifest if item["title"] == "gazelle_nn_base_and_options")

    result = service.read_material(candidate_id=candidate_id, focus="engine horsepower diesel", max_segments=2, collection_id="gaz")

    assert result["candidate_id"] == candidate_id
    assert result["excerpts"]
    assert "149 hp" in result["excerpts"][0]["excerpt"].lower()
    assert result["metadata"]["artifact_chunk_count"] >= 1


def test_search_sales_materials_uses_artifact_text(gaz_runtime_fixture):
    service, manifest = gaz_runtime_fixture
    expected_candidate = next(item["candidate_id"] for item in manifest if item["title"] == "gazelle_nn_base_and_options")

    result = service.search_sales_materials(
        query="turbodiesel horsepower",
        intent="specs",
        families=["gazelle nn"],
        collection_id="gaz",
    )

    assert result["candidate_count"] >= 1
    assert result["candidates"][0]["candidate_id"] == expected_candidate


def test_get_branch_pack_uses_cached_artifacts(gaz_runtime_fixture):
    service, manifest = gaz_runtime_fixture
    expected_candidate = next(item["candidate_id"] for item in manifest if item["title"] == "gazelle_nn_base_and_options")

    result = service.get_branch_pack(
        branch="configuration",
        slots={"transport_type": "cargo", "decision_criterion": "configuration"},
        problem_summary="Need base options and engine guidance for city delivery van.",
        top_k=3,
        collection_id="gaz",
    )

    assert result["candidates"]
    assert result["candidates"][0]["candidate_id"] == expected_candidate
