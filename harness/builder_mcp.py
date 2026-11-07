"""MCP stdio: только декларативное построение новой детали и выбор своего результата."""

import argparse
import base64
import fcntl
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import BadZipFile, ZipFile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness import budget  # noqa: E402
from harness.native_worker import atomic  # noqa: E402
from harness.recipe import validate  # noqa: E402

NATIVE_BUILD_TIMEOUT = 600

TOOLS = [
    {
        "name": "health",
        "description": "Проверить SSH и подключение к уже запущенному КОМПАС. "
        "Не создаёт, не открывает и не изменяет CAD-документы. Вызывать перед build.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True, "destructiveHint": False},
    },
    {
        "name": "view_input",
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False},
        "description": "Показать исходный PNG или его фрагмент. "
        "crop=[x,y,width,height], пиксели от левого верхнего угла. Чужие файлы недоступны.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "crop": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "build",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "description": "Построить новую деталь по полному JSON-рецепту. См. API.md. "
        "Поддерживает variables, expr, constrain и bindings управляющих размеров. "
        "Каждый вызов начинает с пустой детали. Возвращает измерения, параметры CAD и PNG своей модели.",
        "inputSchema": {
            "type": "object",
            "properties": {"recipe": {"type": "object"}},
            "required": ["recipe"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
        "description": "Выбрать успешную попытку этого запуска как окончательную M3D.",
        "inputSchema": {
            "type": "object",
            "properties": {"attempt": {"type": "integer", "minimum": 1}},
            "required": ["attempt"],
            "additionalProperties": False,
        },
    },
]


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def remote_command(settings: dict, directory: Path, *, check_only: bool = False) -> list[str]:
    relative = directory.relative_to(REPO).as_posix().replace("/", "\\")
    root = settings["windows_repo"]
    arguments = [
        settings["windows_python"],
        "-I",
        "-B",
        "-X",
        "utf8",
        root + r"\harness\native_worker.py",
        "--directory",
        root + "\\" + relative,
        "--bin-dir",
        settings["kompas_bin_dir"],
    ]
    if check_only:
        arguments.append("--check")
    script = (
        (
            "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; "
            "$p=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('kStudy.exe','KOMPAS.exe')}); "
            "if ($p.Count -ne 1) {throw 'Expected one running KOMPAS'}; "
            "if ((Split-Path $p[0].ExecutablePath) -ne " + ps_quote(settings["kompas_bin_dir"]) + ") "
            "{throw 'Running KOMPAS differs from configured bin'}; & "
        )
        + " ".join(ps_quote(v) for v in arguments)
        + "; exit $LASTEXITCODE"
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode()
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        settings["ssh_host"],
        "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand " + encoded,
    ]


def verify_bundle(run: Path) -> None:
    manifest = json.loads((run / "control" / "bundle.json").read_text())
    for relative, expected in manifest.items():
        path = REPO / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Trusted bundle changed: {relative}; prepare again")


def connection_failed_before_execution(completed: subprocess.CompletedProcess) -> bool:
    """Только однозначный отказ соединения SSH, до старта удалённой команды."""
    return (
        completed.returncode == 255
        and not completed.stdout
        and re.fullmatch(
            rb"ssh: connect to host [^\r\n ]+ port [0-9]+: "
            rb"(?:Operation not permitted|Permission denied|Connection refused|"
            rb"Connection timed out|No route to host)\r?\n?",
            completed.stderr,
        )
        is not None
    )


def extract_preview(model: Path, destination: Path) -> None:
    """Достать встроенный снимок M3D без вызовов KsAPI."""
    with ZipFile(model) as archive:
        entry = archive.getinfo("Preview")
        if entry.file_size > 4_000_000:
            raise ValueError("M3D preview exceeds size limit")
        data = archive.read(entry)
    offset = data.index(b"II*\x00", 0, 128)
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "preview.tiff"
        source.write_bytes(data[offset:])
        output = Path(tmp) / "preview.png"
        subprocess.run(
            ["/usr/bin/sips", "-s", "format", "png", str(source), "--out", str(output)],
            capture_output=True,
            check=True,
            timeout=15,
        )
        png = output.read_bytes()
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("M3D preview did not decode as PNG")
        atomic(destination, png)


class Bridge:
    def __init__(self, run: Path) -> None:
        self.run = run.resolve(strict=True)
        self.settings = json.loads((run / "control" / "settings.json").read_text())

    def call(self, name: str, args: dict) -> dict:
        if not isinstance(args, dict):
            raise ValueError("Arguments must be an object")
        with (self.run / "control" / "bridge.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify_bundle(self.run)
            if name == "health" and not args:
                return self.health()
            if name == "view_input" and set(args) <= {"crop"}:
                return self.view_input(args.get("crop"))
            if budget.limits(self.run) and (REPO / "runs/hw01-quarantine.json").exists():
                raise RuntimeError("An unresolved CAD call requires operator inspection")
            uncertain = self.run / "control" / "uncertain.json"
            if uncertain.exists():
                state = json.loads(uncertain.read_text())
                if (
                    state.get("reason") == "cleanup_failed_after_save"
                    and name == "finish"
                    and set(args) == {"attempt"}
                    and args["attempt"] == state["attempt"]
                ):
                    return self.finish(args["attempt"])
                raise RuntimeError("Previous native call unresolved; operator must inspect it")
            if name == "build" and set(args) == {"recipe"}:
                budget.reserve_build(self.run)
                return self.build(validate(args["recipe"]))
            if name == "finish" and set(args) == {"attempt"}:
                return self.finish(args["attempt"])
            raise ValueError("Unsupported tool or arguments")

    def health(self) -> dict:
        completed = subprocess.run(
            remote_command(self.settings, self.run / "control", check_only=True),
            input=b"",
            capture_output=True,
            timeout=45,
            check=False,
        )
        if completed.returncode:
            if budget.limits(self.run):
                atomic(self.run / "control/infrastructure-error.json", b'{"stage":"health"}')
            if connection_failed_before_execution(completed):
                message = (
                    "SSH connection blocked before Windows execution. Call this tool as a registered "
                    "MCP server, not by spawning it inside the agent's sandboxed shell. "
                    "Operator: inspect SSH/host permissions; do not enable network for the builder."
                )
            else:
                message = "Native preflight failed: " + completed.stderr.decode(errors="replace")[-2000:]
            return {"isError": True, "content": [{"type": "text", "text": message}]}
        report = json.loads(completed.stdout)
        report["build_quarantined"] = (self.run / "control" / "uncertain.json").exists()
        return {"content": [{"type": "text", "text": json.dumps(report, ensure_ascii=False)}]}

    def view_input(self, crop: Any) -> dict:
        source = self.run / "builder" / "input" / "drawing.png"
        data = source.read_bytes()
        if crop is not None:
            if not (
                isinstance(crop, list)
                and len(crop) == 4
                and all(type(v) is int and 0 <= v <= 20000 for v in crop)
                and crop[2] > 0
                and crop[3] > 0
            ):
                raise ValueError("crop: [x,y,width,height], bounded nonnegative integers")
            width, height = int.from_bytes(data[16:20]), int.from_bytes(data[20:24])
            x, y, w, h = crop
            if x + w > width or y + h > height:
                raise ValueError(f"Crop outside drawing {width}x{height}")
            destination = self.run / "control" / "input-crop.png"
            subprocess.run(
                [
                    "/usr/bin/sips",
                    "-c",
                    str(h),
                    str(w),
                    "--cropOffset",
                    str(y),
                    str(x),
                    str(source),
                    "--out",
                    str(destination),
                ],
                capture_output=True,
                check=True,
                timeout=15,
            )
            data = destination.read_bytes()
        return {
            "content": [{"type": "image", "mimeType": "image/png", "data": base64.b64encode(data).decode()}]
        }

    def build(self, recipe: dict) -> dict:
        budget.check(self.run)
        recipe = validate(recipe)
        raw = self.run / "raw"
        attempts = [int(p.name.split("-")[1]) for p in raw.glob("attempt-*") if p.is_dir()]
        attempt = max(attempts, default=0) + 1
        if attempt > 30:
            raise RuntimeError("Attempt limit reached")
        directory = raw / f"attempt-{attempt:04d}"
        directory.mkdir()
        payload = json.dumps(recipe, ensure_ascii=False).encode()
        atomic(directory / "recipe.json", payload)
        atomic(self.run / "control/native-inflight.json", json.dumps({"attempt": attempt}).encode())
        try:
            completed = subprocess.run(
                remote_command(self.settings, directory),
                input=payload,
                capture_output=True,
                timeout=NATIVE_BUILD_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            budget.quarantine(self.run, "Native call timed out")
            atomic(directory / "ssh.stdout", exc.stdout or b"")
            atomic(directory / "ssh.stderr", exc.stderr or b"")
            atomic(self.run / "control" / "uncertain.json", json.dumps({"attempt": attempt}).encode())
            raise RuntimeError(
                "Native call timed out. Further builds blocked; do not retry automatically"
            ) from exc
        (self.run / "control/native-inflight.json").unlink(missing_ok=True)
        atomic(directory / "ssh.stdout", completed.stdout)
        atomic(directory / "ssh.stderr", completed.stderr)
        atomic(directory / "transport.json", json.dumps({"returncode": completed.returncode}).encode())
        result_path = directory / "result.json"
        if completed.returncode != 0 or not result_path.is_file():
            if connection_failed_before_execution(completed) and not result_path.exists():
                if budget.limits(self.run):
                    atomic(self.run / "control/infrastructure-error.json", b'{"stage":"build_transport"}')
                return {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "attempt": attempt,
                                    "status": "transport_failed_before_execution",
                                    "message": "SSH failed before execution. No CAD operation started. "
                                    "Use registered MCP tools, not a bridge subprocess in the agent shell. "
                                    "Call health after the operator repairs connectivity.",
                                }
                            ),
                        }
                    ],
                }
            atomic(self.run / "control" / "uncertain.json", json.dumps({"attempt": attempt}).encode())
            budget.quarantine(self.run, "Native worker did not complete normally")
            raise RuntimeError("Native worker did not complete normally; operator inspection required")
        result = json.loads(result_path.read_text())
        result["attempt"] = attempt
        preview = directory / "preview.png"
        if result["status"] == "built" and result.get("cleanup_error"):
            budget.quarantine(self.run, "Cleanup failed after saving a model")
            atomic(
                self.run / "control" / "uncertain.json",
                json.dumps({"attempt": attempt, "reason": "cleanup_failed_after_save"}).encode(),
            )
        if result["status"] == "built" and not preview.is_file():
            try:
                extract_preview(directory / "model.m3d", preview)
                result.pop("preview_error", None)
                result["preview_source"] = "embedded_m3d"
            except (BadZipFile, KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
                result["preview_error"] = str(exc)
        content: list[dict] = [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
        if preview.is_file():
            content.append(
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": base64.b64encode(preview.read_bytes()).decode(),
                }
            )
        return {"content": content, "isError": result["status"] != "built"}

    def finish(self, attempt: Any) -> dict:
        budget.check(self.run)
        if type(attempt) is not int or not 1 <= attempt <= 30:
            raise ValueError("Invalid attempt number")
        directory = self.run / "raw" / f"attempt-{attempt:04d}"
        result = json.loads((directory / "result.json").read_text())
        model = directory / "model.m3d"
        data = model.read_bytes()
        if result["status"] != "built" or hashlib.sha256(data).hexdigest() != result["model_sha256"]:
            raise ValueError("Attempt is not a verified build")
        if budget.limits(self.run) and (self.run / "artifacts/selection.json").exists():
            raise RuntimeError("Final selection is immutable")
        budget.check(self.run)
        atomic(self.run / "artifacts" / "model.m3d", data)
        atomic(
            self.run / "artifacts" / "selection.json",
            json.dumps(
                {
                    "attempt": attempt,
                    "model_sha256": result["model_sha256"],
                    "status": "selected_for_evaluation",
                    "selected_at": datetime.now(UTC).isoformat(),
                }
            ).encode(),
        )
        return {"content": [{"type": "text", "text": "M3D сохранена; оценка выполняется отдельно."}]}


def dispatch(bridge: Bridge, message: dict) -> dict | None:
    method = message.get("method")
    if "id" not in message:
        return None
    response = {"jsonrpc": "2.0", "id": message["id"]}
    if method == "initialize":
        response["result"] = {
            "protocolVersion": message["params"]["protocolVersion"],
            "serverInfo": {"name": "isolated-kompas-builder", "version": "2"},
            "capabilities": {"tools": {}},
            "instructions": "Сначала вызови health, затем build и finish. Это ограниченный мост к КОМПАС. "
            "Вызывай его как MCP-инструмент чата. Не запускай сервер через subprocess в shell агента: "
            "такой процесс наследует запрет сети и SSH не сработает. Доступ к эталонам и сырой SSH не нужны.",
        }
    elif method == "ping":
        response["result"] = {}
    elif method == "tools/list":
        response["result"] = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            params = message["params"]
            response["result"] = bridge.call(params["name"], params.get("arguments", {}))
        except Exception as exc:  # noqa: BLE001 -- граница протокола, ошибка видна агенту
            response["result"] = {
                "isError": True,
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            }
    else:
        response["error"] = {"code": -32601, "message": "Method not available"}
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    bridge = Bridge(args.run)
    while line := sys.stdin.buffer.readline(1_000_002):
        if len(line) > 1_000_000 or not line.endswith(b"\n"):
            raise SystemExit("MCP request too large")
        response = dispatch(bridge, json.loads(line))
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
