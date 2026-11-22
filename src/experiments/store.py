"""Неизменяемое хранилище серий на диске: снимок условий, расписание и попытки."""

import importlib.metadata
import random
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.core.config import ExperimentConfig
from src.core.utils import atomic, digest, read_json, safe_name, sha256, write_json

VERIFIED_PACKAGES = (
    "openai-agents",
    "openai",
    "pydantic",
    "pillow",
    "cadquery",
    "trimesh",
    "mlflow",
    "numpy",
    "scipy",
    "rtree",
    "manifold3d",
    "matplotlib",
    "pyyaml",
)


def study_path(root: Path, name: str) -> Path:
    path = root / "runs" / safe_name(name)
    if path.is_symlink():
        raise ValueError("Папка серии не должна быть символической ссылкой")
    return path


def _snapshot_files(root: Path) -> list[Path]:
    src = root / "src"
    files = sorted(src.rglob("*.py")) if src.is_dir() else []
    for extra in (root / "pyproject.toml", root / "uv.lock"):
        if extra.is_file():
            files.append(extra)
    return files


def freeze(
    root: Path,
    name: str,
    config: ExperimentConfig,
    manifest: dict,
    split: str,
    pairs: list[str] | None,
    repeats: int,
) -> tuple[Path, dict]:
    if repeats < 1 or repeats > 100:
        raise ValueError("Число повторов должно быть от 1 до 100")
    selected = [item for item in manifest["items"] if item["split"] == split]
    if pairs:
        unknown = set(pairs) - {item["pair_id"] for item in selected}
        if unknown:
            raise ValueError(f"Деталь отсутствует в выбранной выборке {split}: {', '.join(sorted(unknown))}")
        selected = [item for item in selected if item["pair_id"] in pairs]
    if not selected:
        raise ValueError("Пустая выборка деталей")

    snapshot_files = _snapshot_files(root)
    source_hashes = {path.relative_to(root).as_posix(): sha256(path) for path in snapshot_files}
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in VERIFIED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    identity = {
        "config": config.model_dump(mode="json"),
        "manifest": manifest,
        "source_sha256": source_hashes,
        "split": split,
        "pairs": [item["pair_id"] for item in selected],
        "repeats": repeats,
        "versions": versions,
    }
    directory = study_path(root, name)
    metadata = directory / "study.json"
    if metadata.exists():
        existing = verify_study(directory)
        if existing["fingerprint"] != digest(identity):
            raise ValueError("Конфиг, промпты, код, окружение или выборка изменились: задайте новый --study")
        return directory, existing
    if directory.exists():
        raise ValueError("Папка серии уже существует без study.json; используйте новое имя --study")

    schedule = [
        {"pair_id": item["pair_id"], "repeat": repeat, "variant": variant}
        for item in selected
        for repeat in range(1, repeats + 1)
        for variant in config.variants
    ]
    random.Random(config.execution.order_seed).shuffle(schedule)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    ).stdout.strip()
    record = {
        **identity,
        "fingerprint": digest(identity),
        "name": name,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit or None,
        "schedule": schedule,
        "formal_validation": (
            split == "val"
            and repeats == config.execution.repeats
            and len(selected) == config.execution.validation_size
            and not pairs
        ),
    }
    write_json(metadata, record)
    atomic(directory / "study.sha256", (sha256(metadata) + "\n").encode("utf-8"))
    return directory, record


def verify_study(directory: Path) -> dict:
    metadata = directory / "study.json"
    checksum = directory / "study.sha256"
    if not metadata.is_file() or not checksum.is_file():
        raise ValueError(f"Отсутствует снимок серии в {directory}")
    if sha256(metadata) != checksum.read_text(encoding="utf-8").strip():
        raise ValueError("Снимок серии study.json был изменён")
    return read_json(metadata)


def case_name(pair_id: str, repeat: int) -> str:
    return f"{safe_name(pair_id)}-r{repeat:02d}"


def attempts(parent: Path, case: str) -> list[Path]:
    return sorted(parent.glob(f"{case}-a[0-9][0-9][0-9]"))


def read_record(directory: Path) -> dict:
    path = directory / "record.json"
    record = read_json(path)
    if record["status"] != "started":
        checksum_path = directory / "record.sha256"
        if not checksum_path.is_file() or sha256(path) != checksum_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Журнал попытки изменён или не завершён корректно: {directory.name}")
        for name, expected in record.get("files", {}).items():
            target = directory / name
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or target.is_symlink()
                or not target.resolve().is_relative_to(directory.resolve())
            ):
                raise ValueError(f"Артефакт попытки изменён: {directory.name}/{name}")
            if not target.is_file():
                if Path(name).parts[:1] == ("debug",):
                    continue
                raise ValueError(f"Артефакт попытки изменён: {directory.name}/{name}")
            if sha256(target) != expected:
                raise ValueError(f"Артефакт попытки изменён: {directory.name}/{name}")
    return record


def begin_attempt(parent: Path, case: str, retry_infrastructure: bool) -> tuple[Path, dict | None]:
    if parent.is_symlink():
        raise ValueError("Папка попыток не должна быть символической ссылкой")
    previous = attempts(parent, case)
    if previous:
        latest = previous[-1]
        record = read_record(latest) if (latest / "record.json").is_file() else {"status": "started"}
        failed_infrastructure = (
            record["status"] == "started" or record.get("failure_kind") == "infrastructure"
        )
        if not (retry_infrastructure and failed_infrastructure):
            return latest, record
    number = len(previous) + 1
    if number > 999:
        raise ValueError("Превышен лимит технических попыток (999)")
    directory = parent / f"{case}-a{number:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory, None


def save_record(directory: Path, record: dict, *, final: bool = False) -> None:
    if (directory / "record.sha256").exists():
        raise ValueError("Завершённую попытку нельзя перезаписать")
    if final:
        record["files"] = {
            file.relative_to(directory).as_posix(): sha256(file)
            for file in sorted(directory.rglob("*"))
            if file.is_file() and file.name not in {"record.json", "record.sha256"}
        }
    write_json(directory / "record.json", record)
    if final:
        atomic(directory / "record.sha256", (sha256(directory / "record.json") + "\n").encode("utf-8"))


def purge_study_files(root: Path, name: str) -> list[str]:
    """Удаляет локальные директории серии (runs/, results/, logs/) и её файл блокировки."""
    safe = safe_name(name)
    root_resolved = root.resolve()
    removed: list[str] = []
    targets = (
        root / "runs" / safe,
        root / "results" / safe,
        root / "logs" / safe,
        root / "logs/.locks" / f"{safe}.lock",
    )
    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        if target.is_symlink() or not target.resolve().is_relative_to(root_resolved):
            raise ValueError(f"Небезопасный путь при удалении серии: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed.append(target.relative_to(root).as_posix())
    return removed
