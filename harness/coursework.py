"""ДЗ1: проверка комплекта, изолированная генерация и повтор оценки без КОМПАС."""

import argparse
import hashlib
import json
import math
import shutil
import sys
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.native_worker import atomic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/hw01/baseline.toml"


@contextmanager
def cad_session() -> Iterator[None]:
    """Не допускает параллельных генераций и оценки в операторском CLI на Mac."""
    import fcntl

    path = ROOT / ".hw01-work/cad.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Path escapes bundle: {relative}")
    return path


def configuration(path: Path) -> dict:
    config = tomllib.loads(path.read_text())
    for name in ("max_seconds", "max_builds"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if config["max_builds"] > 30:
        raise ValueError("Bridge supports at most 30 builds")
    for name in ("dataset", "bundle"):
        inside(ROOT, config[name])
    if not config["experiment_id"].replace("-", "").isalnum():
        raise ValueError("Invalid experiment ID")
    return config


def verify_files(root: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = inside(root, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Missing or changed artifact: {relative}")


def check(config: dict, *, reviewed: bool = False) -> dict:
    dataset = inside(ROOT, config["dataset"])
    inventory = read(dataset / "manifest.json")
    if (dataset / "SHA256.json").exists():
        verify_files(dataset, read(dataset / "SHA256.json"))
    items = inventory["items"]
    ids = [item["pair_id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate pair ID")
    splits: dict[str, set[str]] = {}
    for item in items:
        if item["split"] not in {"dev", "val"}:
            raise ValueError("Unknown split")
        splits.setdefault(item["family"], set()).add(item["split"])
        for field in ("input_png", "reference_m3d", "source_cdw"):
            if record := item.get(field):
                verify_files(dataset, {record["path"]: record["sha256"]})
        if reviewed and item["split"] == "val" and item["pair_validation"] != "passed":
            raise ValueError(f"Pair review pending or failed: {item['pair_id']}")
    if any(len(split) > 1 for split in splits.values()):
        raise ValueError("Related geometry occurs in dev and val")
    if (
        sum(item["split"] == "dev" for item in items) != 1
        or sum(item["split"] == "val" for item in items) != 2
    ):
        raise ValueError("HW01 requires one dev and two validation cases")
    verify_files(ROOT, read(ROOT / "eval/SOURCE.json")["files"])
    bundle = inside(ROOT, config["bundle"])
    if (bundle / "SHA256.json").exists():
        verify_files(bundle, read(bundle / "SHA256.json"))
    return inventory


def experiment_files(config_path: Path, config: dict) -> dict[str, str]:
    paths = [
        config_path,
        ROOT / config["dataset"] / "manifest.json",
        ROOT / "eval/SOURCE.json",
        ROOT / "uv.lock",
    ]
    paths += list((ROOT / "harness").glob("*.py"))
    paths += [ROOT / config["dataset"] / "SHA256.json", ROOT / "pyproject.toml"]
    paths += [ROOT / "scripts" / name for name in ("ksapi_mcp_ssh.sh", "ksapi_mcp_stdio.py")]
    paths += [
        ROOT / "harness/prompts" / f"{config[key]}.md"
        for key in ("prompt_version", "api_version", "connection_version")
    ]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def freeze(config_path: Path, config: dict) -> None:
    target = config_path.with_name("freeze.json")
    current = experiment_files(config_path, config)
    if target.exists():
        if read(target)["files"] != current:
            raise ValueError("Frozen experiment changed; use a new experiment version")
    else:
        write(
            target,
            {
                "experiment_id": config["experiment_id"],
                "frozen_at": datetime.now(UTC).isoformat(),
                "files": current,
            },
        )


def ledger_path(config: dict) -> Path:
    return ROOT / ".hw01-work" / config["experiment_id"] / "attempts.json"


def generate(config_path: Path, config: dict, split: str, settings: Path) -> None:
    from harness import run as runner

    inventory = check(config, reviewed=True)
    path = ledger_path(config)
    ledger = read(path) if path.exists() else {"dev": {}, "val": {}}
    if split == "val":
        if not ledger["dev"] or any(
            not (ROOT / p / "control/exit.json").exists() for p in ledger["dev"].values()
        ):
            raise ValueError("Finish the dev run before validation")
        freeze(config_path, config)
    elif config_path.with_name("freeze.json").exists():
        raise ValueError("Development is closed for this frozen experiment")
    if (ROOT / "runs/hw01-quarantine.json").exists():
        raise RuntimeError("Inspect unresolved CAD call before starting another generation")
    for item in inventory["items"]:
        if item["split"] != split:
            continue
        key = item["pair_id"]
        if key in ledger[split]:
            existing = ROOT / ledger[split][key]
            if (existing / "control/exit.json").exists():
                print(f"Retained original attempt: {existing}", flush=True)
                continue
            runner.require_unstarted(existing)
            run = existing
        else:
            name = f"{config['experiment_id']}-{key.lower()}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
            run = ROOT / "runs/e2e" / name
            runner.prepare(run, settings, dataset=ROOT / config["dataset"], pair_id=key, experiment=config)
            ledger[split][key] = run.relative_to(ROOT).as_posix()
            write(path, ledger)
        print(f"Generate {key}: {run}", flush=True)
        runner.launch(run, config["model"])
        if (ROOT / "runs/hw01-quarantine.json").exists():
            raise RuntimeError("Native operation needs inspection; subsequent cases have not started")


def model_input(directory: Path, root: Path) -> dict:
    return {
        "native": (directory / "model.m3d").relative_to(root).as_posix(),
        "mesh": (directory / "model.stl").relative_to(root).as_posix(),
        "facts": (directory / "eval-facts.json").relative_to(root).as_posix(),
        "export_manifest": (directory / "export.json").relative_to(root).as_posix(),
        "units": "mm",
    }


def copy_generation(run: Path, destination: Path) -> dict:
    manifest = read(run / "manifest.json")
    if not (run / "control/exit.json").exists():
        raise ValueError(f"Unfinished generation: {run.name}")
    status = manifest["status"]
    if status == "generated_pending_evaluation":
        verify_files(run, {"artifacts/model.m3d": manifest["selection"]["model_sha256"]})
        outcome = "success"
    elif status == "timeout":
        outcome = "timeout"
    elif status == "no_selected_model":
        if manifest.get("returncode") or (run / "control/infrastructure-error.json").exists():
            raise ValueError("Infrastructure failure cannot be scored as failed generation")
        outcome = "failed"
    else:
        raise ValueError(f"Infrastructure/interrupted run is not a zero score: {status}")
    budget_file = run / "control/budget.json"
    record = {
        key: manifest.get(key)
        for key in (
            "run_id",
            "pair_id",
            "model",
            "model_settings",
            "prompt_version",
            "prompt_sha256",
            "limits",
            "started_at",
            "finished_at",
            "elapsed_seconds",
            "returncode",
            "timed_out",
            "selection",
        )
    }
    record.update(
        generation_status=outcome,
        build_calls=read(budget_file)["build_calls"] if budget_file.exists() else None,
    )
    record["raw_sha256"] = {
        p.relative_to(run).as_posix(): sha(p) for p in sorted((run / "raw").rglob("*")) if p.is_file()
    }
    record["isolation"] = read(run / "control/isolation-check.json")
    events = run / "raw/events.jsonl"
    record["usage"] = []
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "turn.completed" and event.get("usage"):
                record["usage"].append(event["usage"])
    record["cost_money"] = None
    # Публичная производная запись не содержит локальных путей оператора.
    record["isolation"]["stderr"] = record["isolation"]["stderr"].replace(str(ROOT), "<workspace>")
    write(destination / "generation.json", record)
    for name in ("recipe.json", "report.md", "model_plan.md"):
        source = run / "builder/work" / name
        if source.exists():
            atomic(destination / name, source.read_bytes().replace(str(ROOT).encode(), b"<workspace>"))
    return record


def prepare_evaluation(config: dict) -> Path:
    inventory = check(config, reviewed=True)
    ledger = read(ledger_path(config))
    work = ROOT / ".hw01-work" / config["experiment_id"] / "evaluation"
    if (work / "evaluation.json").exists():
        return work
    cases = []
    for item in inventory["items"]:
        if item["split"] != "val":
            continue
        key = item["pair_id"]
        if key not in ledger["val"]:
            raise ValueError(f"Missing planned attempt: {key}")
        run = ROOT / ledger["val"][key]
        record = copy_generation(run, work / "generation" / key)
        reference = work / "inputs" / key / "reference"
        shutil.copytree(ROOT / config["dataset"] / "qa" / key, reference, dirs_exist_ok=True)
        case = {
            "id": key,
            "family": item["family"],
            "generation_status": record["generation_status"],
            "reference": model_input(reference, work),
        }
        if record["generation_status"] == "success":
            predicted = work / "inputs" / key / "prediction"
            predicted.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / "artifacts/model.m3d", predicted / "model.m3d")
            case["prediction"] = model_input(predicted, work)
        cases.append(case)
    write(
        work / "evaluation.json",
        {"purpose": "benchmark", "config": config["evaluation"], "export": config["export"], "cases": cases},
    )
    return work


def export_bundle(config: dict, work: Path, output: Path) -> None:
    from eval.geometry.pipeline import load_evaluation

    bundle = ROOT / config["bundle"]
    if bundle.exists():
        raise FileExistsError("Published bundle is immutable; choose a new experiment version")
    temporary = bundle.with_name(bundle.name + ".preparing")
    temporary.mkdir(parents=True, exist_ok=False)
    evaluation = load_evaluation(work).model_dump()
    for case in evaluation["cases"]:
        for role in ("reference", "prediction"):
            model = case.get(role)
            if not model:
                continue
            destination = temporary / "inputs" / case["id"] / role
            destination.mkdir(parents=True, exist_ok=True)
            export = (work / model["export_manifest"]).parent
            for name in (
                "model.stl",
                "eval-facts.json",
                "kernel-facts.json",
                "raw-facts.json",
                "export-source.json",
                "export.json",
            ):
                shutil.copyfile(export / name, destination / name)
            shutil.copyfile(work / model["native"], destination / "model.m3d")
            case[role] = model_input(destination, temporary)
    if evaluation["boolean_results"]:
        shutil.copyfile(work / evaluation["boolean_results"], temporary / "boolean-results.json")
        evaluation["boolean_results"] = "boolean-results.json"
    shutil.copytree(work / "generation", temporary / "generation")
    write(temporary / "evaluation.json", evaluation)
    shutil.copyfile(output / "metrics.json", temporary / "expected-metrics.json")
    shutil.copyfile(output / "report.md", temporary / "metrics-report.md")
    shutil.copytree(output / "raw", temporary / "metric-evidence")
    write(
        temporary / "SHA256.json",
        {p.relative_to(temporary).as_posix(): sha(p) for p in sorted(temporary.rglob("*")) if p.is_file()},
    )
    temporary.rename(bundle)


def compare_metrics(expected: dict, actual: dict) -> None:
    if not expected["summary"]["rankable"] or not actual["summary"]["rankable"]:
        raise ValueError("Primary metrics are incomplete")
    a = {row["id"]: row for row in expected["cases"]}
    b = {row["id"]: row for row in actual["cases"]}
    if a.keys() != b.keys():
        raise ValueError("Case list changed")
    for key in a:
        for metric in ("solid_iou", "f_score"):
            left, right = a[key]["primary"][metric]["value"], b[key]["primary"][metric]["value"]
            if left is None or right is None or not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Replay differs: {key}/{metric}: {left} vs {right}")


def evaluate(config_path: Path, config: dict, settings_path: Path | None) -> Path:
    from eval.geometry.backend import BackendSettings
    from eval.geometry.io import create_artifact_directory
    from eval.geometry.pipeline import execute
    from eval.geometry.progress import progress_display

    check(config, reviewed=True)
    if not config_path.with_name("freeze.json").exists():
        raise ValueError("No frozen experiment; run validation first")
    freeze(config_path, config)
    backend = None
    if settings_path:
        if (ROOT / "runs/hw01-quarantine.json").exists():
            raise RuntimeError("Inspect pending CAD operation before evaluation")
        work = prepare_evaluation(config)
        settings = read(settings_path)
        backend = BackendSettings(
            transport="ssh",
            ssh_host=settings["ssh_host"],
            remote_repo=settings["windows_repo"],
            python=settings["windows_python"],
            bin_dir=settings["kompas_bin_dir"],
        )
    else:
        bundle = ROOT / config["bundle"]
        verify_files(bundle, read(bundle / "SHA256.json"))
        work = create_artifact_directory(ROOT / ".hw01-work/replays", "replay")
        shutil.copytree(bundle, work, dirs_exist_ok=True)
    with progress_display() as progress:
        output, _ = execute(work, offline=settings_path is None, backend=backend, progress=progress)
    result = read(output / "metrics.json")
    if not result["summary"]["rankable"]:
        raise RuntimeError(f"Incomplete evaluation: {output}")
    if settings_path:
        export_bundle(config, work, output)
    else:
        compare_metrics(read(work / "expected-metrics.json"), result)
    print(output, flush=True)
    return output


def report(config: dict) -> str:
    check(config, reviewed=True)
    bundle = ROOT / config["bundle"]
    metrics = read(bundle / "expected-metrics.json")
    lines = [
        "# Baseline ДЗ1",
        "",
        "## Кратко",
        "",
        "Прямой агент; один запуск на деталь, 15 минут и до 15 построений.",
        "",
        "| Деталь | IoU | F-score | Время, с | Вызовы build | Статус |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in metrics["cases"]:
        record = read(bundle / "generation" / case["id"] / "generation.json")
        values = [case["primary"][name]["value"] for name in ("solid_iou", "f_score")]
        formatted = ["не считалось" if v is None else f"{v:.6f}" for v in values]
        lines.append(
            f"| {case['id']} | {' | '.join(formatted)} | {record['elapsed_seconds']:.1f} "
            f"| {record['build_calls']} | {record['generation_status']} |"
        )
    lines += ["", "Средние по проверочным деталям:"]
    for name in ("solid_iou", "f_score"):
        value = metrics["summary"]["primary"][name]["mean"]
        lines.append(f"- {name}: {value}")
    lines += [
        "",
        f"VSR: {metrics['summary']['valid_shape_rate']}; IR: {metrics['summary']['invalidity_ratio']}.",
        "",
        "Пилот на двух объектах; устойчивость по повторным запускам "
        "и качество на всём корпусе не оценивались.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "generate", "evaluate", "report"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("dev", "val"))
    parser.add_argument(
        "--settings", type=Path, help="Локальные настройки Windows; evaluate без них работает офлайн"
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = configuration(config_path)
    if args.action == "check":
        inventory = check(config, reviewed=True)
        print(f"Verified {inventory['dataset_id']}: dataset, evaluator and available artifacts")
    elif args.action == "generate":
        if args.split is None or args.settings is None:
            parser.error("generate requires --split and --settings")
        with cad_session():
            generate(config_path, config, args.split, args.settings)
    elif args.action == "evaluate":
        if args.settings:
            with cad_session():
                evaluate(config_path, config, args.settings)
        else:
            evaluate(config_path, config, None)
    else:
        print(report(config), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"HW01: {error}", file=sys.stderr)
        raise SystemExit(2) from error
