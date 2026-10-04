from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from .contracts import IsmartGenerationConfig


def _clean_payload(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {str(key): _clean_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clean_payload(item) for item in value)
    return value


WORKER_CODE = r"""
import builtins
import io
import json
import os
import sys
import tempfile
import traceback


def _truncate(value, max_chars):
    text = _clean_text(value)
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _clean_text(value):
    return str(value or "").encode("utf-8", "replace").decode("utf-8")


def _exception_from_stderr(stderr):
    lines = [line.strip() for line in str(stderr or "").splitlines() if line.strip()]
    if not lines:
        return "", "", ""
    last = lines[-1]
    import re
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning)?):\s*(.*)$", last)
    if match:
        return match.group(1), match.group(2), last
    for line in reversed(lines):
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning)?):\s*(.*)$", line)
        if match:
            return match.group(1), match.group(2), line
    return "", "", last


def _run(req):
    code = _clean_text(req.get("code") or "")
    stdin_value = _clean_text(req.get("stdin") or "")
    max_chars = int(req.get("max_output_chars") or 4000)
    old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
    old_cwd = os.getcwd()
    out, err = io.StringIO(), io.StringIO()
    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="ismart_python_run_") as tmp_dir:
        try:
            os.chdir(tmp_dir)
            sys.stdin = io.StringIO(stdin_value)
            sys.stdout = out
            sys.stderr = err
            namespace = {"__name__": "__main__", "__builtins__": builtins.__dict__}
            try:
                exec(compile(code, "<ismart_sandbox>", "exec"), namespace, namespace)
            except SystemExit as exc:
                raw_code = exc.code
                if raw_code in (None, 0):
                    exit_code = 0
                elif isinstance(raw_code, int):
                    exit_code = raw_code
                else:
                    exit_code = 1
                    print(raw_code, file=sys.stderr)
            except BaseException:
                exit_code = 1
                traceback.print_exc()
        finally:
            sys.stdin, sys.stdout, sys.stderr = old_stdin, old_stdout, old_stderr
            os.chdir(old_cwd)
    stdout = _truncate(out.getvalue(), max_chars)
    stderr = _truncate(err.getvalue(), max_chars)
    exception_type, exception_message, last_error_line = _exception_from_stderr(stderr)
    return {
        "status": "completed",
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "last_error_line": last_error_line,
    }


for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get("op") == "shutdown":
            print(json.dumps({"status": "shutdown"}, ensure_ascii=True), flush=True)
            break
        if request.get("op") == "version":
            response = {"status": "completed", "exit_code": 0, "stdout": sys.version.split()[0] + "\n", "stderr": ""}
        else:
            response = _run(request)
    except BaseException as exc:
        response = {
            "status": "worker_error",
            "exit_code": 1,
            "stdout": "",
            "stderr": traceback.format_exc(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "last_error_line": str(exc),
        }
    print(json.dumps(response, ensure_ascii=True), flush=True)
"""


class PythonSandbox:
    def __init__(self, config: IsmartGenerationConfig) -> None:
        self.command = tuple(config.python_sandbox_command)
        self.timeout_seconds = float(config.python_sandbox_timeout_seconds)
        self.max_output_chars = int(config.python_sandbox_max_output_chars)
        self._root = tempfile.TemporaryDirectory(prefix="ismart_python_sandbox_")
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process and process.poll() is None:
                try:
                    self._send({"op": "shutdown"})
                except Exception:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
            self._process = None
        self._root.cleanup()

    def python_version(self) -> str:
        result = self.run_code("", "", op="version")
        return str(result.get("stdout") or "").strip()

    def run_code(self, code: str, stdin: str = "", *, op: str = "run") -> dict[str, Any]:
        with self._lock:
            try:
                process = self._ensure_process()
            except FileNotFoundError as exc:
                return {
                    "status": "command_not_found",
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "error": str(exc),
                    "command": list(self.command),
                }
            except OSError as exc:
                return {
                    "status": "process_start_error",
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "error": str(exc),
                    "command": list(self.command),
                }
            try:
                self._send(
                    {
                        "op": op,
                        "code": code,
                        "stdin": stdin,
                        "max_output_chars": self.max_output_chars,
                    }
                )
            except (BrokenPipeError, OSError):
                self._restart_locked()
                try:
                    process = self._ensure_process()
                except OSError as exc:
                    return {
                        "status": "process_start_error",
                        "exit_code": None,
                        "stdout": "",
                        "stderr": "",
                        "error": str(exc),
                        "command": list(self.command),
                    }
                self._send(
                    {
                        "op": op,
                        "code": code,
                        "stdin": stdin,
                        "max_output_chars": self.max_output_chars,
                    }
                )
            try:
                response = self._responses.get(timeout=max(0.1, self.timeout_seconds))
            except queue.Empty:
                self._restart_locked()
                return {
                    "status": "timeout",
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timeout_seconds": self.timeout_seconds,
                }
            if response is None:
                stderr = self._read_process_stderr(process)
                self._restart_locked()
                return {
                    "status": "worker_exited",
                    "exit_code": process.poll(),
                    "stdout": "",
                    "stderr": stderr,
                }
            response["command"] = list(self.command)
            return response

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process and self._process.poll() is None:
            return self._process
        self._start_locked()
        if self._process is None:
            raise RuntimeError("Python sandbox process was not started.")
        return self._process

    def _start_locked(self) -> None:
        self._drain_responses()
        command = [*self.command, "-I", "-B", "-u", "-c", WORKER_CODE]
        env = self._env()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self._root.name,
            env=env,
        )
        self._reader = threading.Thread(target=self._read_stdout, args=(self._process,), daemon=True)
        self._reader.start()

    def _restart_locked(self) -> None:
        process = self._process
        if process and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        self._process = None
        self._start_locked()

    def _send(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Python sandbox process stdin is unavailable.")
        self._process.stdin.write(json.dumps(_clean_payload(payload), ensure_ascii=True) + "\n")
        self._process.stdin.flush()

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            self._responses.put(None)
            return
        for line in process.stdout:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = {"status": "protocol_error", "stdout": "", "stderr": line}
            self._responses.put(data)
        self._responses.put(None)

    def _drain_responses(self) -> None:
        while True:
            try:
                self._responses.get_nowait()
            except queue.Empty:
                return

    def _read_process_stderr(self, process: subprocess.Popen[str]) -> str:
        if process.stderr is None:
            return ""
        try:
            return process.stderr.read() or ""
        except Exception:
            return ""

    def _env(self) -> dict[str, str]:
        env: dict[str, str] = {"PYTHONIOENCODING": "utf-8"}
        for key in ("PATH", "SystemRoot", "SYSTEMROOT", "TEMP", "TMP"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def __enter__(self) -> "PythonSandbox":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class DisabledPythonSandbox:
    def python_version(self) -> str:
        return ""

    def run_code(self, code: str, stdin: str = "", *, op: str = "run") -> dict[str, Any]:
        return {"status": "disabled", "exit_code": None, "stdout": "", "stderr": ""}

    def close(self) -> None:
        return None
