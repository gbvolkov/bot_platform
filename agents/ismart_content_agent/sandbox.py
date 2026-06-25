from __future__ import annotations

import subprocess
import sys
import os
from typing import Any

from pydantic import ValidationError

from .contracts import ContentItem, ExecutableSolution, SandboxResult


def run_sandbox(item: ContentItem, *, timeout_seconds: int = 3) -> SandboxResult:
    if item.template_id != "practice_python":
        return SandboxResult(content_id=item.content_id, status="not_applicable")

    raw_solution = item.service_payload.get("service_solution")
    tests = item.service_payload.get("tests")
    try:
        solution = ExecutableSolution.model_validate(raw_solution)
    except ValidationError:
        return SandboxResult(
            content_id=item.content_id,
            status="fail",
            reason="practice_python requires an executable service_solution object",
        )
    if not isinstance(tests, list) or not tests:
        return SandboxResult(
            content_id=item.content_id,
            status="fail",
            reason="practice_python requires tests in service_payload",
        )
    code = solution.code

    results: list[dict[str, Any]] = []
    for test in tests:
        stdin = str(test.get("stdin", ""))
        expected = str(test.get("expected_stdout", ""))
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", "-c", code],
                input=stdin,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                content_id=item.content_id,
                status="timeout",
                tests=results,
                reason=f"Timeout on test {test.get('test_id')}",
            )
        if completed.returncode != 0:
            return SandboxResult(
                content_id=item.content_id,
                status="runtime_error",
                tests=results,
                reason=f"Python service check failed on test {test.get('test_id')}",
            )
        actual = _normalize_stdout(completed.stdout)
        expected_normalized = _normalize_stdout(expected)
        passed = actual == expected_normalized
        results.append(
            {
                "test_id": test.get("test_id"),
                "passed": passed,
            }
        )
        if not passed:
            return SandboxResult(
                content_id=item.content_id,
                status="fail",
                tests=results,
                reason=f"Output mismatch on test {test.get('test_id')}",
            )
    return SandboxResult(content_id=item.content_id, status="pass", tests=results)


def _normalize_stdout(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = "\n".join(line.rstrip() for line in lines)
    return normalized.rstrip("\n") + ("\n" if normalized else "")
