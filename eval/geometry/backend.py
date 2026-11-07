"""Транспорт вычислителя: локальный Windows или SSH с общей папкой репозитория."""

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import Field, model_validator

from eval.geometry.io import atomic_json, atomic_text, create_artifact_directory, read_json, sha256
from eval.geometry.models import DTO
from eval.geometry.readiness import REPO, worker_fingerprint


class BackendSettings(DTO):
    transport: Literal["local", "ssh"] = "local"
    bin_dir: str
    python: str = sys.executable
    ssh_host: str | None = None
    remote_repo: str | None = None
    timeout_seconds: int = Field(default=900, ge=1)

    @model_validator(mode="after")
    def check_remote(self) -> "BackendSettings":
        if self.transport == "ssh" and (not self.ssh_host or not self.remote_repo):
            raise ValueError("SSH требует ssh_host и remote_repo с общей папкой репозитория")
        return self


def settings_from_path(path: Path | None) -> BackendSettings | None:
    if path is not None:
        return BackendSettings.model_validate(read_json(path))
    bin_dir = os.environ.get("KOMPAS_BIN_DIR")
    if os.name == "nt" and bin_dir:
        return BackendSettings(bin_dir=bin_dir)
    return None


def powershell_command(arguments: list[str]) -> str:
    quoted = " ".join("'" + value.replace("'", "''") + "'" for value in arguments)
    command = "$ErrorActionPreference='Stop'; & " + quoted + "; exit $LASTEXITCODE"
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand " + encoded


class WorkerClient:
    def __init__(self, settings: BackendSettings, directory: Path) -> None:
        self.settings = settings
        self.directory = directory
        self.available: bool | None = None
        self.failure: str | None = None

    def mapped(self, path: Path) -> str:
        if self.settings.transport == "local":
            return str(path.resolve())
        relative = path.resolve().relative_to(REPO)
        return str(PureWindowsPath(self.settings.remote_repo) / PureWindowsPath(relative.as_posix()))

    def invoke(self, action: str, payload: dict[str, Any], sources: dict[str, Path]) -> tuple[Path, dict]:
        if self.available is False:
            raise RuntimeError(self.failure or "Вычислитель недоступен")
        job = create_artifact_directory(self.directory, action)
        inputs = {}
        for name, source in sources.items():
            expected = sha256(source)
            target = job / f"{name}.m3d"
            temporary = target.with_suffix(".m3d.tmp")
            shutil.copyfile(source, temporary)
            if sha256(temporary) != expected:
                raise ValueError(f"Модель изменилась во время копирования: {source}")
            temporary.replace(target)
            inputs[name] = {"path": target.name, "sha256": expected}
        request = {"action": action, "inputs": inputs, "implementation": worker_fingerprint(), **payload}
        atomic_json(job / "request.json", request)
        worker = REPO / "eval/geometry/kompas_worker.py"
        arguments = [
            self.settings.python,
            "-I",
            "-B",
            "-X",
            "utf8",
            self.mapped(worker),
            self.mapped(job),
            self.settings.bin_dir,
        ]
        command = arguments
        if self.settings.transport == "ssh":
            command = [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                self.settings.ssh_host,
                powershell_command(arguments),
            ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=self.settings.timeout_seconds, check=False
            )
            atomic_text(job / "stdout.log", completed.stdout)
            atomic_text(job / "stderr.log", completed.stderr)
            if not (job / "response.json").is_file():
                raise RuntimeError(f"Вычислитель не вернул ответ (код {completed.returncode}); {job}")
            response = read_json(job / "response.json")
            if completed.returncode or response.get("status") != "ok":
                raise RuntimeError(response.get("reason", f"Код вычислителя {completed.returncode}"))
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
            self.available = False
            self.failure = str(error)
            atomic_json(job / "transport-error.json", {"reason": str(error)})
            raise RuntimeError(str(error)) from error
        self.available = True
        return job, response

    def health(self) -> dict:
        return self.invoke("health", {}, {})[1]
