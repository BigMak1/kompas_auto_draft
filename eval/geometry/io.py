"""Воспроизводимый ввод и атомарная запись артефактов."""

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def create_artifact_directory(parent: Path, prefix: str, started_at: datetime | None = None) -> Path:
    """Атомарно резервирует читаемое имя; совпавшая секунда получает номер повтора."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", prefix):
        raise ValueError("Префикс папки должен состоять из латиницы, цифр, дефисов и подчёркиваний")
    started_at = started_at or datetime.now().astimezone()
    name = f"{prefix}_{started_at:%Y-%m-%d_%H-%M-%S}"
    attempt = 1
    while True:
        path = parent / (name if attempt == 1 else f"{name}_{attempt}")
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            attempt += 1
        else:
            return path


def create_evaluation_directory(parent: Path, started_at: datetime) -> Path:
    return create_artifact_directory(parent, "eval", started_at)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, data: Any) -> None:
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
