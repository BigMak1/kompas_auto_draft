"""Подготовка/проверка/явный запуск изолированного эксперимента KF0011."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness import budget
from harness.builder_mcp import NATIVE_BUILD_TIMEOUT, remote_command, verify_bundle
from harness.native_worker import atomic

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs" / "e2e" / "kf0011-codex-003"
CODEX_VERSION = "codex-cli 0.166.0"


def prepare(
    run: Path,
    settings_path: Path,
    *,
    dataset: Path | None = None,
    pair_id: str = "KF0011",
    experiment: dict | None = None,
) -> None:
    """Чистый комплект выбранной пары: только вход и общие инструкции."""
    if run.parent != REPO / "runs" / "e2e" or not re.fullmatch(r"[a-z0-9-]+", run.name):
        raise ValueError("New run must be a named directory directly under runs/e2e")
    if run.exists():
        raise RuntimeError("Run already exists; never overwrite a previous experiment")
    dataset = dataset or REPO / "kompas_data" / "e2e-27-v1"
    experiment = experiment or {}
    inventory = json.loads((dataset / "manifest.json").read_text())
    item = next(item for item in inventory["items"] if item["pair_id"] == pair_id)
    data = (dataset / item["input_png"]["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != item["input_png"]["sha256"]:
        raise RuntimeError("Frozen input checksum mismatch")
    settings = json.loads(settings_path.read_text())
    temporary = run.with_name(run.name + ".preparing")
    temporary.mkdir(parents=True, exist_ok=False)
    for path in ("builder/input", "builder/library", "builder/work", "control", "raw", "artifacts"):
        (temporary / path).mkdir(parents=True)
    atomic(temporary / "builder/input/drawing.png", data)
    for source, target in (
        (experiment.get("prompt_version", "editable_builder_v2") + ".md", "AGENTS.md"),
        (experiment.get("api_version", "api_v2") + ".md", "API.md"),
        (experiment.get("connection_version", "connection_v2") + ".md", "CONNECTION.md"),
    ):
        atomic(temporary / "builder" / target, (REPO / "harness/prompts" / source).read_bytes())
    for name in ("kompas_lib.py", "AGENTS.md", "README.md", "llms.txt"):
        if (REPO / "vendor/kompas_lib" / name).exists():
            shutil.copyfile(REPO / "vendor/kompas_lib" / name, temporary / "builder/library" / name)
    atomic(temporary / "builder/.builder-root", b"")
    atomic(temporary / "control/canary.txt", b"Evaluator-only isolation sentinel\n")
    atomic(temporary / "control/settings.json", json.dumps(settings, indent=2).encode())
    manifest = {
        "schema_version": 1,
        "run_id": run.name,
        "status": "prepared",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": inventory.get("dataset_id", dataset.name),
        "pair_id": pair_id,
        "purpose": "editable_cad_development_run",
        "recipe_version": 2,
        "input": item["input_png"] | {"path": "builder/input/drawing.png"},
        "reference": {"access": "evaluation_only", **item["reference_m3d"]},
        "model": None,
        "model_settings": None,
        "cost": None,
        "metrics": None,
        "prompt": "builder/AGENTS.md",
        "prompt_version": experiment.get("prompt_version", "editable_builder_v2"),
        "prompt_sha256": hashlib.sha256((temporary / "builder/AGENTS.md").read_bytes()).hexdigest(),
        "command_template": (
            f".venv/bin/python -m harness.run launch --run runs/e2e/{run.name} --model <model-id>"
        ),
        "output": "artifacts/model.m3d",
        "schema_used": "harness/recipe.py",
        "library_source": {"path": "vendor/kompas_lib", "copied_unmodified": True},
        "isolation": "must_pass_preflight_before_launch",
        "evaluation": "pending",
    }
    if experiment:
        manifest["limits"] = {key: experiment[key] for key in ("max_seconds", "max_builds")}
        manifest["reasoning_effort"] = experiment["reasoning_effort"]
        manifest["purpose"] = "hw01_baseline"
    atomic(temporary / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    temporary.rename(run)
    seal(run)


def require_unstarted(run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        manifest["status"] != "prepared"
        or (run / "control/started.json").exists()
        or any((run / "raw").glob("attempt-*"))
        or (run / "artifacts/selection.json").exists()
    ):
        raise RuntimeError("Run is no longer empty and prepared; create a new experiment")


def toml(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(json.dumps(k) + "=" + toml(v) for k, v in value.items()) + "}"
    return json.dumps(value, ensure_ascii=False)


def configuration(run: Path) -> dict:
    builder = run / "builder"
    manifest = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
    disabled = (
        "apps",
        "connectors",
        "plugins",
        "remote_plugin",
        "browser_use",
        "computer_use",
        "memories",
        "memory_tool",
        "multi_agent",
        "js_repl",
        "code_mode",
        "hooks",
        "codex_hooks",
        "plugin_hooks",
        "request_permissions",
        "request_permissions_tool",
        "request_rule",
        "view_image",
        "workspace_dependencies",
        "tool_suggest",
        "skill_search",
        "shell_snapshot",
        "shell_snapshot_v2",
        "remote_control",
        "mcp_connection_sharing",
        "daemon_auto_start",
    )
    return {
        "default_permissions": "kf0011",
        "permissions.kf0011.filesystem": {
            "/": "read",
            "/Users": "deny",
            "/private": "deny",
            "/Volumes": "deny",
            "/Network": "deny",
            "/home": "deny",
            "/cores": "deny",
            "/tmp": "deny",
            "/var": "deny",
            "/private/var/select": "read",
            str(builder): "read",
            str(builder / "work"): "write",
        },
        "permissions.kf0011.network.enabled": False,
        "approval_policy": "never",
        "web_search": "disabled",
        "allow_login_shell": False,
        "project_root_markers": [".builder-root"],
        "project_doc_max_bytes": 0,
        "features": dict.fromkeys(disabled, False) | {"skip_host_skill_discovery": True},
        "skills.bundled.enabled": False,
        "skills.include_instructions": False,
        "skills.config": [],
        "model_reasoning_effort": manifest.get("reasoning_effort", "high"),
        "shell_environment_policy.inherit": "none",
        "shell_environment_policy.set": {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "en_US.UTF-8",
            "TMPDIR": str(builder / "work"),
        },
        "mcp_servers": {
            "kompas-builder": {
                "command": str(REPO / ".venv" / "bin" / "python"),
                "args": ["-I", "-B", str(REPO / "harness" / "builder_mcp.py"), "--run", str(run)],
                "required": True,
                "startup_timeout_sec": 30,
                "tool_timeout_sec": NATIVE_BUILD_TIMEOUT + 30,
                "enabled_tools": ["health", "build", "finish", "view_input"],
                # Пользователь поручил построение в этом run; разрешены только операции
                # ограниченного моста. Shell/сеть/файлы остаются под прежней песочницей.
                "tools": {
                    name: {"approval_mode": "approve"} for name in ("health", "build", "finish", "view_input")
                },
            },
            "ksapi-expert": {
                "command": "/bin/bash",
                "args": [str(REPO / "scripts" / "ksapi_mcp_ssh.sh")],
                "required": True,
                "startup_timeout_sec": 180,
                "tool_timeout_sec": 120,
                "enabled_tools": ["search", "fetch", "sources", "grep_ksapi_wrappers"],
            },
        },
    }


def overrides(config: dict) -> list[str]:
    return [arg for key, value in config.items() for arg in ("-c", key + "=" + toml(value))]


def seal(run: Path) -> None:
    """Обновить хеши после осознанной правки ещё не начатого комплекта оператором."""
    require_unstarted(run)
    manifest = json.loads((run / "manifest.json").read_text())
    png = run / "builder" / "input" / "drawing.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != manifest["input"]["sha256"]:
        raise RuntimeError("Input PNG differs from frozen manifest")
    atomic(run / "control" / "codex-config.json", json.dumps(configuration(run), indent=2).encode())
    files = list((REPO / "harness").glob("*.py")) + list((REPO / "vendor" / "kompas_lib").glob("*"))
    files += [REPO / "scripts" / name for name in ("ksapi_mcp_ssh.sh", "ksapi_mcp_stdio.py")]
    files += [
        run / "control" / "settings.json",
        run / "control" / "codex-config.json",
    ]
    files += [
        p
        for p in (run / "builder").rglob("*")
        if p.is_file() and "work" not in p.relative_to(run / "builder").parts
    ]
    bundle = {
        p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)
    }
    atomic(run / "control" / "bundle.json", json.dumps(bundle, indent=2).encode())


def native_check(run: Path) -> dict:
    verify_bundle(run)
    settings = json.loads((run / "control" / "settings.json").read_text())
    result = subprocess.run(
        remote_command(settings, run / "control", check_only=True),
        input=b"",
        capture_output=True,
        timeout=40,
        check=False,
    )
    atomic(
        run / "control" / "native-check-result.json",
        json.dumps(
            {
                "checked_at": datetime.now(UTC).isoformat(),
                "returncode": result.returncode,
                "stdout": result.stdout.decode(errors="replace"),
                "stderr": result.stderr.decode(errors="replace"),
            },
            ensure_ascii=False,
            indent=2,
        ).encode(),
    )
    if result.returncode:
        raise RuntimeError("Native preflight failed: " + result.stderr.decode(errors="replace")[-2000:])
    report = json.loads((run / "control" / "native-preflight.json").read_text())
    if report["status"] != "read_only_preflight_passed":
        raise RuntimeError("Native runtime unavailable")
    return report


def preflight(run: Path) -> dict:
    verify_bundle(run)
    version = subprocess.check_output(["codex", "--version"], text=True).strip()
    if version != CODEX_VERSION:
        raise RuntimeError(f"Revalidate isolation for {version}; expected {CODEX_VERSION}")
    builder = run / "builder"
    if any(p.is_symlink() for p in builder.rglob("*")):
        raise RuntimeError("Builder workspace contains symlinks")
    config = configuration(run)
    # Тот же профиль, который передаётся codex exec; модель/сеть OpenAI не вызываются.
    probe = """import errno, json, pathlib, socket
b = pathlib.Path.cwd()
results = {}
results['input_readable'] = (b/'input/drawing.png').read_bytes()[:8] == b'\\x89PNG\\r\\n\\x1a\\n'
for key, path in [('sibling_denied', b.parent/'control/canary.txt'),
                  ('repo_denied', b.parents[3]/'README.md'),
                  ('source_denied', b.parents[4]/'kompas-drawings/README.md'),
                  ('reference_denied', b.parents[3]/'kompas_data/e2e-27-v1/references/KF0011/model.m3d'),
                  ('previous_run_denied', b.parent.parent/'kf0011-codex-001/artifacts/model.m3d'),
                  ('public_reference_denied', b.parents[3]/'data/hw01/v1/references/KF0017/model.m3d')]:
    try:
        path.read_bytes()
        results[key] = False
    except PermissionError:
        results[key] = True
link = b/'work/escape-probe'
try:
    link.symlink_to(b.parent/'control/canary.txt')
    try:
        link.read_bytes()
        results['symlink_denied'] = False
    except PermissionError:
        results['symlink_denied'] = True
finally:
    link.unlink(missing_ok=True)
try:
    with (b/'API.md').open('ab'):
        pass
    results['instructions_readonly'] = False
except PermissionError:
    results['instructions_readonly'] = True
p = b/'work/write-probe'
p.write_text('ok')
results['work_writable'] = p.read_text() == 'ok'
p.unlink()
s = socket.socket()
s.settimeout(2)
try:
    results['network_denied'] = s.connect_ex(('10.211.55.3', 22)) in (errno.EPERM, errno.EACCES)
finally:
    s.close()
print(json.dumps(results))
raise SystemExit(0 if all(results.values()) else 1)
"""
    command = [
        "codex",
        "sandbox",
        "-P",
        "kf0011",
        *overrides(config),
        "-C",
        str(builder),
        "--",
        "/usr/bin/python3",
        "-c",
        probe,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    report = {
        "codex_version": version,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    atomic(run / "control" / "isolation-check.json", json.dumps(report, indent=2).encode())
    if result.returncode != 0:
        raise RuntimeError("Isolation preflight failed; see control/isolation-check.json")
    checks = json.loads(result.stdout)
    if len(checks) != 11 or not all(value is True for value in checks.values()):
        raise RuntimeError("Isolation preflight incomplete")
    return report


def launch(run: Path, model: str) -> int:
    require_unstarted(run)
    if (REPO / "runs/hw01-quarantine.json").exists():
        raise RuntimeError("Unresolved CAD operation; inspect runs/hw01-quarantine.json")
    preflight(run)
    native_check(run)
    config = configuration(run)
    command = [
        "codex",
        "--no-daemon",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--ephemeral",
        "--skip-git-repo-check",
        *overrides(config),
        "--model",
        model,
        "-C",
        str(run / "builder"),
        "--image",
        str(run / "builder" / "input" / "drawing.png"),
        "--json",
        "--output-last-message",
        str(run / "raw" / "final-response.md"),
        "-",
    ]
    prompt = (run / "builder" / "AGENTS.md").read_bytes()
    start = {
        "model_requested": model,
        "command": command,
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "status": "started",
        "evaluation": "pending",
        "started_at": datetime.now(UTC).isoformat(),
    }
    atomic(run / "control" / "started.json", json.dumps(start, ensure_ascii=False, indent=2).encode())
    manifest = json.loads((run / "manifest.json").read_text())
    manifest.update(
        status="running",
        model=model,
        model_settings={"reasoning_effort": config["model_reasoning_effort"]},
        started_at=start["started_at"],
        command=command,
        prompt_sha256=start["prompt_sha256"],
    )
    atomic(run / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    # Удалить служебные переменные родительской сессии. Авторизация Codex остаётся штатной.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CODEX_THREAD")}
    budget.begin(run)
    started_clock = time.monotonic()
    timed_out = False
    try:
        with (
            (run / "raw" / "events.jsonl").open("xb") as events,
            (run / "raw" / "stderr.log").open("xb") as errors,
        ):
            with subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=events, stderr=errors, env=env
            ) as process:
                try:
                    process.communicate(input=prompt, timeout=manifest.get("limits", {}).get("max_seconds"))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if (run / "control/native-inflight.json").exists():
                        budget.quarantine(
                            run, "Deadline reached during native operation; inspect before another run"
                        )
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                result = subprocess.CompletedProcess(command, process.returncode)
    except BaseException:
        manifest.update(status="interrupted_or_launcher_failed", finished_at=datetime.now(UTC).isoformat())
        atomic(run / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode())
        raise
    selected = run / "artifacts" / "selection.json"
    manifest.update(
        status="generated_pending_evaluation"
        if selected.exists()
        else "timeout"
        if timed_out
        else "no_selected_model",
        elapsed_seconds=time.monotonic() - started_clock,
        timed_out=timed_out,
        returncode=result.returncode,
        finished_at=datetime.now(UTC).isoformat(),
    )
    if selected.exists():
        manifest["selection"] = json.loads(selected.read_text())
    atomic(run / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    atomic(
        run / "control" / "exit.json",
        json.dumps(
            {"returncode": result.returncode, "model_exists": (run / "artifacts" / "model.m3d").is_file()}
        ).encode(),
    )
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "preflight", "native-check", "seal", "launch", "print-config"]
    )
    parser.add_argument("--model")
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--settings", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == "prepare":
        if not args.settings:
            parser.error("prepare requires operator --settings JSON")
        prepare(run, args.settings)
        print(f"Prepared {run}; run preflight and native-check")
    elif args.action == "print-config":
        print(json.dumps(configuration(run), ensure_ascii=False, indent=2))
    elif args.action == "preflight":
        print(json.dumps(preflight(run), ensure_ascii=False, indent=2))
    elif args.action == "native-check":
        print(json.dumps(native_check(run), ensure_ascii=False, indent=2))
    elif args.action == "seal":
        seal(run)
        print("Bundle hashes updated; run preflight again before launch")
    else:
        if not args.model:
            parser.error("launch requires an explicit --model")
        raise SystemExit(launch(run, args.model))


if __name__ == "__main__":
    main()
