"""Переносимая серия: зафиксированная матрица, снимок кода и неизменяемые попытки."""

import hashlib
import importlib.metadata
import io
import json
import random
import re
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from src.experiments.config import ExperimentConfig
from src.utils import atomic, read_json, sha256, write_json


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Идентификатор: латинские буквы, цифры, дефис или подчёркивание")
    return name


def study_path(root: Path, name: str) -> Path:
    path = root / "run" / safe_name(name)
    if path.is_symlink():
        raise ValueError("Серия не должна быть символической ссылкой")
    return path


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
        raise ValueError("Повторов должно быть от 1 до 100")
    selected = [item for item in manifest["items"] if item["split"] == split]
    if pairs:
        if set(pairs) - {item["pair_id"] for item in selected}:
            raise ValueError("Деталь отсутствует в выбранной части набора")
        selected = [item for item in selected if item["pair_id"] in pairs]
    if not selected:
        raise ValueError("Пустая выборка")
    snapshot_files = [*sorted((root / "src").rglob("*.py")), root / "pyproject.toml", root / "uv.lock"]
    source_hashes = {p.relative_to(root).as_posix(): sha256(p) for p in snapshot_files}
    versions = {"python": sys.version.split()[0]}
    for package in (
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
    ):
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
        existing = read_json(metadata)
        if existing["fingerprint"] != digest(identity):
            raise ValueError("Конфиг, код, окружение или выборка изменились: задайте новый --study")
        verify_study(directory)
        return directory, existing
    if directory.exists():
        raise ValueError("Папка серии существует без study.json; используйте новое имя")
    schedule = [
        {"pair_id": item["pair_id"], "repeat": repeat, "variant": variant}
        for item in selected
        for repeat in range(1, repeats + 1)
        for variant in config.variants
    ]
    random.Random(config.order_seed).shuffle(schedule)
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
        "formal_validation": split == "val"
        and repeats == config.repeats
        and len(selected) == config.validation_size
        and set(config.variants) == {"direct", "passport_only", "passport_png", "notes_png"},
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in snapshot_files:
            archive.writestr(file.relative_to(root).as_posix(), file.read_bytes())
    atomic(directory / "source.zip", buffer.getvalue())
    record["source_archive_sha256"] = sha256(directory / "source.zip")
    write_json(metadata, record)
    atomic(directory / "study.sha256", (sha256(metadata) + "\n").encode())
    return directory, record


def verify_study(directory: Path) -> dict:
    if sha256(directory / "study.json") != (directory / "study.sha256").read_text().strip():
        raise ValueError("Снимок серии изменён")
    study = read_json(directory / "study.json")
    if sha256(directory / "source.zip") != study["source_archive_sha256"]:
        raise ValueError("Архив исходного кода изменён")
    return study


def case_name(pair: str, repeat: int) -> str:
    return f"{safe_name(pair)}-r{repeat:02d}"


def attempts(parent: Path, case: str) -> list[Path]:
    return sorted(parent.glob(f"{case}-a[0-9][0-9][0-9]"))


def read_record(directory: Path) -> dict:
    path = directory / "record.json"
    record = read_json(path)
    if record["status"] != "started":
        if sha256(path) != (directory / "record.sha256").read_text().strip():
            raise ValueError(f"Журнал попытки изменён: {directory.name}")
        for name, expected in record.get("files", {}).items():
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or (directory / name).is_symlink()
                or not (directory / name).resolve().is_relative_to(directory.resolve())
                or sha256(directory / name) != expected
            ):
                raise ValueError(f"Артефакт изменён: {directory.name}/{name}")
    return record


def begin_attempt(parent: Path, case: str, retry: bool) -> tuple[Path, dict | None]:
    if parent.is_symlink():
        raise ValueError("Папка попыток не должна быть символической ссылкой")
    previous = attempts(parent, case)
    if previous:
        latest = previous[-1]
        record = read_record(latest) if (latest / "record.json").is_file() else {"status": "started"}
        failed_infrastructure = (
            record["status"] == "started" or record.get("failure_kind") == "infrastructure"
        )
        if not (retry and failed_infrastructure):
            return latest, record
    number = len(previous) + 1
    if number > 999:
        raise ValueError("Слишком много технических попыток")
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
        atomic(directory / "record.sha256", (sha256(directory / "record.json") + "\n").encode())
