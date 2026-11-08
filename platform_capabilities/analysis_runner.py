"""Bounded Docker execution. This launcher is never available to generated code."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

import httpx


class AnalysisUnavailable(RuntimeError):
    pass


class DockerAnalysisRunner:
    def __init__(self, image: str = "bot-platform-analysis:mentor-v1", *,
                 staging_root: str | None = None, docker: str = "docker", timeout: float = 30):
        self.image, self.docker, self.timeout = image, docker, timeout
        self.staging_root = staging_root

    async def identity(self) -> str:
        rc, output = await self._command("image", "inspect", "--format", "{{.Id}}", self.image)
        if rc or not output.strip():
            raise AnalysisUnavailable("Private analysis image is not available locally")
        return output.strip()

    async def _command(self, *args: str, limit: int = 16384, timeout: float = 10) -> tuple[int, str]:
        try:
            process = await asyncio.create_subprocess_exec(self.docker, *args,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        except OSError as exc:
            raise AnalysisUnavailable("Docker launcher is unavailable") from exc
        chunks = bytearray()
        async def collect():
            while block := await process.stdout.read(4096):
                if len(chunks) + len(block) > limit:
                    raise ValueError("Analysis output exceeds its bounded protocol limit")
                chunks.extend(block)
            await process.wait()
        try:
            await asyncio.wait_for(collect(), timeout)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return process.returncode, chunks.decode("utf-8", "replace")

    async def run(self, code: str, files: dict[str, bytes]) -> dict:
        if len(code.encode()) > 65536 or sum(map(len, files.values())) > 32 * 1024 * 1024:
            raise ValueError("Analysis job exceeds input limit")
        environment = await self.identity()
        name = "mentor-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="mentor-analysis-", dir=self.staging_root) as temporary:
            # TemporaryDirectory is host-private (0700 on Linux). Mount a
            # readable child, so UID 65534 can traverse /inputs without making
            # the host job directory or its parent publicly traversable.
            root = Path(temporary).resolve() / "inputs"
            root.mkdir()
            root.chmod(0o755)
            for filename, data in files.items():
                if Path(filename).name != filename or filename in {".", "..", "program.py"}:
                    raise ValueError("Invalid job input filename")
                target = root / filename
                target.write_bytes(data)
                target.chmod(0o444)
            program = root / "program.py"
            program.write_text(code, encoding="utf-8")
            program.chmod(0o444)
            # Adapt the prototype's JSON result boundary. Collect the file while the
            # container is alive: Docker discards tmpfs mounts after process exit.
            wrapper = "import json,runpy,contextlib; from pathlib import Path\nwith open('/output/stdout.txt','w') as out, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):\n runpy.run_path('/inputs/program.py',run_name='__main__')\np=Path('/output/result.json')\nif p.stat().st_size>16384: raise ValueError('Result exceeds 16 KiB')\nprint(json.dumps(json.loads(p.read_text()),allow_nan=False))\n"
            try:
                rc, diagnostic = await self._command("create", "--pull", "never", "--name", name, "--network", "none",
                    "--read-only", "--user", "65534:65534", "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges", "--pids-limit", "64", "--cpus", "2",
                    "--memory", "1g", "--memory-swap", "1g", "--ulimit", "fsize=16777216:16777216",
                    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
                    "--tmpfs", "/output:rw,noexec,nosuid,size=16m,mode=1777",
                    "--mount", f"type=bind,source={root},target=/inputs,readonly",
                    "--workdir", "/output", "--log-driver", "none", environment,
                    "python", "-I", "-B", "-u", "-c", wrapper)
                if rc:
                    raise AnalysisUnavailable("Private container creation failed: " + diagnostic[:2000])
                rc, diagnostic = await self._command("start", "--attach", name, timeout=self.timeout)
                if rc:
                    return {"exit_code": rc, "stderr": diagnostic, "result": None}
                try:
                    result = json.loads(diagnostic)
                except json.JSONDecodeError:
                    return {"exit_code": 1, "stderr": "Invalid bounded JSON result: " + diagnostic[:2000], "result": None}
                return {"exit_code": 0, "stderr": "", "result": result,
                        "environment": environment, "code_hash": hashlib.sha256(code.encode()).hexdigest()}
            finally:
                # Container kill/removal owns termination, including cancellation of its CLI wrapper.
                cleanup = asyncio.create_task(self._command("rm", "--force", name))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise


class RemoteAnalysisRunner:
    def __init__(self, url: str, token: str):
        if not token:
            raise ValueError("Private analysis service credential required")
        self.url, self.token = url, token

    async def identity(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(self.url.rstrip("/") + "/identity", headers={"Authorization": "Bearer " + self.token})
                response.raise_for_status()
                return response.json()["environment"]
        except httpx.HTTPError as exc:
            raise AnalysisUnavailable("Private analysis service is unavailable") from exc

    async def run(self, code: str, files: dict[str, bytes]) -> dict:
        job_id = uuid.uuid4().hex
        headers = {"Authorization": "Bearer " + self.token}
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(self.url.rstrip("/") + "/analyze",
                    headers=headers, json={"id": job_id, "code": code,
                        "files": {name: base64.b64encode(data).decode() for name, data in files.items()}})
                response.raise_for_status()
                return response.json()
        except (asyncio.CancelledError, httpx.TimeoutException):
            async def stop():
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.delete(self.url.rstrip("/") + "/analyze/" + job_id, headers=headers)
            await asyncio.shield(stop())
            raise
        except httpx.HTTPError as exc:
            raise AnalysisUnavailable("Private analysis service is unavailable") from exc
