from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.ismart_content_agent.contracts import GenerationRequest
from agents.ismart_content_agent.generators import generate_content_item
from agents.ismart_content_agent.pipeline import (
    generate_run,
    mark_approved,
    mark_preview_passed,
    publish_run,
    validate_run,
)
from agents.ismart_content_agent.publisher import build_sanitized_platform_payload
from agents.ismart_content_agent.run_store import RunStore
from agents.ismart_content_agent.source_reader import resolve_request_sources
from agents.ismart_content_agent.templates import get_template
from agents.ismart_content_agent.validators import find_secret_leaks, validate_items


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ismart" / "templates"


def load_request(name: str) -> GenerationRequest:
    request = GenerationRequest.model_validate_json((FIXTURES / name).read_text(encoding="utf-8"))
    return resolve_request_sources(request)


def write_request(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    target.write_text((FIXTURES / name).read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_generation_request_requires_template_id():
    payload = json.loads((FIXTURES / "theory_request.json").read_text(encoding="utf-8"))
    payload.pop("template_id")

    with pytest.raises(ValidationError):
        GenerationRequest.model_validate(payload)


def test_registry_resolves_mvp_templates_and_rejects_unknown():
    assert get_template("theory").content_type == "theory"
    assert get_template("practice_python").service_fields == ["service_solution", "tests"]
    assert set(get_template("interactive_template").supported_interactive_codes) == {
        "6A",
        "6D",
        "6G",
        "8D",
        "10D",
        "3H",
        "3D",
    }

    with pytest.raises(ValueError, match="Unknown template_id"):
        get_template("missing")  # type: ignore[arg-type]


def test_practice_python_separates_learner_and_service_payloads():
    item = generate_content_item(load_request("practice_python_request.json"))

    assert "service_solution" not in item.learner_payload
    assert "tests" not in item.learner_payload
    assert item.learner_payload["condition"] == "Ввести число, вывести его удвоенное значение (a * 2)."
    assert item.learner_payload["source_binding"]["course_tracker"]["lesson_number"] == 5
    assert "service_solution" in item.service_payload
    assert "tests" in item.service_payload
    solution = item.service_payload["service_solution"]
    assert solution["entrypoint"] == "main.py"
    assert "if __name__ == \"__main__\":" in solution["code"]
    assert not find_secret_leaks(item)


def test_generation_returns_valid_content_item_for_each_template():
    for fixture_name in [
        "theory_request.json",
        "practice_python_request.json",
        "self_study_request.json",
        "control_question_request.json",
        "interactive_6a_request.json",
        "interactive_6d_request.json",
        "interactive_6g_request.json",
        "interactive_3h_request.json",
    ]:
        item = generate_content_item(load_request(fixture_name))
        assert item.content_id
        assert item.learner_payload["course_id"] == "python-basic-8-9"
        assert item.requirement_ids


def test_no_service_leak_validator_blocks_keys_in_learner_payload():
    item = generate_content_item(load_request("practice_python_bad_secret_leak.json"))
    item.learner_payload["service_solution"] = "print('leak')"

    report = validate_items([item])

    assert report.status == "failed"
    assert any(v.code == "SECURITY_NO_SERVICE_LEAK" for v in report.violations)
    criterion = next(c for c in report.criteria if c.criterion_id == "SECURITY_NO_SERVICE_LEAK")
    assert criterion.status == "failed"


def test_sandbox_pass_and_fail_are_deterministic():
    good = generate_content_item(load_request("practice_python_request.json"))
    good_report = validate_items([good])
    assert good_report.status == "passed"
    assert good_report.sandbox_results[0].status == "pass"

    bad = generate_content_item(load_request("practice_python_bad_sandbox.json"))
    bad.service_payload["tests"][0]["expected_stdout"] = "wrong\n"
    bad_report = validate_items([bad])
    assert bad_report.status == "failed"
    assert bad_report.sandbox_results[0].status == "fail"


def test_validation_lists_every_required_criterion():
    item = generate_content_item(load_request("practice_python_request.json"))
    report = validate_items([item])
    criteria = {criterion.criterion_id: criterion for criterion in report.criteria}

    assert set(f"V{number}" for number in range(1, 25)).issubset(criteria)
    assert criteria["V3"].status == "passed"
    assert criteria["V5"].status == "passed"
    assert criteria["V7"].status == "passed"
    assert criteria["V1"].status == "not_applicable"
    assert criteria["PRACT_SOURCE_TASK_MATCH"].status == "passed"
    assert criteria["PRACT_SANDBOX_EXECUTION"].status == "passed"


def test_publisher_builds_sanitized_payload():
    item = generate_content_item(load_request("practice_python_request.json"))
    payload = build_sanitized_platform_payload(item)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "service_solution" not in serialized
    assert "expected_stdout" not in serialized
    assert "answer_key" not in serialized
    assert "tests" not in serialized


def test_pipeline_publish_blocks_non_approved_content(tmp_path):
    request_path = write_request(tmp_path, "theory_request.json")
    run_dir = tmp_path / "run"

    generate_run(request_path, run_dir)
    validate_run(run_dir)

    with pytest.raises(ValueError, match="not approved"):
        publish_run(run_dir)


def test_pipeline_publish_writes_sanitized_manifest_and_platform_payload(tmp_path):
    request_path = write_request(tmp_path, "practice_python_request.json")
    run_dir = tmp_path / "run"

    generate_run(request_path, run_dir)
    validate_run(run_dir)
    item = RunStore(run_dir).read_items()[0]
    mark_approved(run_dir, item.content_id)
    publish_run(run_dir)

    run_document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest = json.dumps(run_document["manifest"], ensure_ascii=False)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    platform_payload = json.dumps(run_document["platform_payloads"], ensure_ascii=False)
    combined = "\n".join([manifest, report, platform_payload])
    assert "service_solution" not in combined
    assert "expected_stdout" not in combined
    assert "answer_key" not in combined
    assert "hidden_tests" not in combined


def test_pipeline_interactive_requires_preview_pass(tmp_path):
    request_path = write_request(tmp_path, "interactive_6a_request.json")
    run_dir = tmp_path / "run"

    generate_run(request_path, run_dir)
    validate_run(run_dir)
    item = RunStore(run_dir).read_items()[0]
    mark_approved(run_dir, item.content_id)

    with pytest.raises(ValueError, match="preview"):
        publish_run(run_dir)

    mark_preview_passed(run_dir, item.content_id, "preview/6a.html")
    publish_run(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["manifest"]
    assert manifest["items"][0]["preview_status"] == "passed"


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agents.ismart_content_agent.cli", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_e2e_non_interactive_publish(tmp_path):
    request_path = write_request(tmp_path, "theory_request.json")
    run_dir = tmp_path / "run"

    assert run_cli("generate", "--request", str(request_path), "--out", str(run_dir), cwd=Path.cwd()).returncode == 0
    assert run_cli("validate", "--run", str(run_dir), cwd=Path.cwd()).returncode == 0
    item = RunStore(run_dir).read_items()[0]
    assert run_cli("approve", "--run", str(run_dir), "--content-id", item.content_id, cwd=Path.cwd()).returncode == 0
    assert run_cli("publish", "--run", str(run_dir), cwd=Path.cwd()).returncode == 0
    run_document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_document["manifest"] is not None


def test_run_store_uses_one_consolidated_json(tmp_path):
    request_path = write_request(tmp_path, "practice_python_request.json")
    run_dir = tmp_path / "run"

    generate_run(request_path, run_dir)
    validate_run(run_dir)

    json_files = list(run_dir.rglob("*.json"))
    assert json_files == [run_dir / "run.json"]
    document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert document["request"]["task_spec"]["task_text"]
    assert document["items"][0]["service_payload"]["service_solution"]["code"]
    assert document["validation"]["criteria"]
    assert document["hitl"]["items"]


def test_cli_e2e_interactive_preview_gate(tmp_path):
    request_path = write_request(tmp_path, "interactive_3h_request.json")
    run_dir = tmp_path / "run"

    assert run_cli("generate", "--request", str(request_path), "--out", str(run_dir), cwd=Path.cwd()).returncode == 0
    assert run_cli("validate", "--run", str(run_dir), cwd=Path.cwd()).returncode == 0
    item = RunStore(run_dir).read_items()[0]
    assert run_cli("approve", "--run", str(run_dir), "--content-id", item.content_id, cwd=Path.cwd()).returncode == 0

    blocked = run_cli("publish", "--run", str(run_dir), cwd=Path.cwd())
    assert blocked.returncode == 1
    assert "preview" in blocked.stderr

    assert (
        run_cli(
            "preview-pass",
            "--run",
            str(run_dir),
            "--content-id",
            item.content_id,
            "--artifact",
            "preview/3h.html",
            cwd=Path.cwd(),
        ).returncode
        == 0
    )
    assert run_cli("publish", "--run", str(run_dir), cwd=Path.cwd()).returncode == 0
