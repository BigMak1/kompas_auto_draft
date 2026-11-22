"""Рабочая SQLite MLflow и проверяемый переносимый снимок для Git."""

import shutil
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from urllib.parse import unquote, urlparse

from filelock import FileLock

from src.utils import read_json, sha256, write_json

# Меняем только поля размещения файлов. Исторические конфиги, тексты и хеши неприкосновенны.
LOCATIONS = (
    ("experiments", "artifact_location", "1=1"),
    ("runs", "artifact_uri", "1=1"),
    ("logged_models", "artifact_location", "1=1"),
    ("trace_tags", "value", "key='mlflow.artifactLocation'"),
    (
        "model_versions",
        "source",
        "name NOT IN (SELECT name FROM registered_model_tags "
        "WHERE key='mlflow.prompt.is_prompt' AND value='true')",
    ),
    (
        "model_versions",
        "storage_location",
        "name NOT IN (SELECT name FROM registered_model_tags "
        "WHERE key='mlflow.prompt.is_prompt' AND value='true')",
    ),
)


def study_lock(root: Path, name: str) -> Path:
    directory = root / "logs/.locks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.lock"


def backup_database(source: Path, destination: Path) -> None:
    """SQLite backup API включает завершённые транзакции из WAL."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src,
        closing(sqlite3.connect(destination)) as dst,
    ):
        src.backup(dst)
        if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Снимок SQLite не прошёл integrity_check")


def rewrite_locations(database: Path, root: Path, *, portable: bool) -> None:
    """Вызывается только на отключённой копии БД, не на работающем сервере."""
    with closing(sqlite3.connect(database)) as connection, connection:
        for table, column, condition in LOCATIONS:
            if not connection.execute(f"PRAGMA table_info({table})").fetchall():
                continue
            rows = connection.execute(f"SELECT rowid, {column} FROM {table} WHERE {condition}").fetchall()
            for rowid, value in rows:
                if table == "experiments" and rowid == 0:
                    occupied = connection.execute(
                        "SELECT count(*) FROM runs WHERE experiment_id=0"
                    ).fetchone()[0]
                    if not occupied:
                        # MLflow создаёт пустой Default с cwd/mlruns; у него нет файлов для переноса.
                        value = "mlflow/files/default"
                if not value or value.startswith(("runs:/", "models:/", "prompts:/")):
                    continue
                parsed = urlparse(value)
                if parsed.scheme not in ("", "file"):
                    raise ValueError(f"Внешнее хранилище нельзя публиковать как локальное: {table}.{column}")
                path = Path(unquote(parsed.path))
                if path.is_absolute():
                    try:
                        relative = path.relative_to(root.resolve())
                    except ValueError as error:
                        raise ValueError(f"Артефакт вне проекта: {value}") from error
                else:
                    relative = path
                if ".." in relative.parts or relative.parts[:2] != ("mlflow", "files"):
                    raise ValueError(f"Неожиданное размещение MLflow: {relative}")
                updated = relative.as_posix() if portable else (root / relative).resolve().as_uri()
                connection.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?", (updated, rowid))


def verify_snapshot(root: Path) -> dict:
    snapshot = read_json(root / "mlflow/snapshot.json")
    if snapshot.get("format") != 1:
        raise ValueError("Неизвестная версия снимка MLflow")
    for name, expected in snapshot["files"].items():
        path = root / "mlflow" / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise ValueError("Недопустимый путь снимка MLflow")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Файл снимка MLflow отсутствует или изменён: {name}")
    return snapshot


def runtime_database(root: Path) -> Path:
    root = root.resolve()
    directory = root / "mlflow/.runtime"
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "tracking.db"
    published = root / "mlflow/tracking.db"
    origin = directory / "origin.json"
    with FileLock(directory / "storage.lock"):
        if published.exists():
            checksum = sha256(published)
            if database.exists():
                saved = read_json(origin) if origin.exists() else {}
                if saved.get("published_sha256") != checksum:
                    raise ValueError(
                        "Опубликованный снимок MLflow обновился. Рабочая история сохранена. "
                        "Для явного восстановления с резервной копией: uv run poe tracking restore"
                    )
            else:
                snapshot = verify_snapshot(root)
                temporary = directory / "initializing.db"
                temporary.unlink(missing_ok=True)
                backup_database(published, temporary)
                rewrite_locations(temporary, root, portable=False)
                for name in snapshot.get("directories", []):
                    relative = Path(name)
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or relative.parts[:2] != ("mlflow", "files")
                    ):
                        raise ValueError("Недопустимая папка снимка MLflow")
                    (root / relative).mkdir(parents=True, exist_ok=True)
                temporary.replace(database)
                write_json(origin, {"published_sha256": checksum})
    return database


def publish(root: Path) -> dict:
    """Не делает commit/push; блокирует наши записи и проверяет каждый публикуемый файл."""
    root = root.resolve()
    database = runtime_database(root)
    if not database.exists():
        raise ValueError("Рабочая история MLflow отсутствует")
    directory = root / "mlflow"
    with ExitStack() as stack:
        for study in sorted((root / "run").glob("*/study.json")):
            stack.enter_context(FileLock(study_lock(root, study.parent.name), timeout=0))
        for name in ("storage", "catalog", "sync"):
            stack.enter_context(FileLock(directory / ".runtime" / f"{name}.lock", timeout=0))
        from mlflow.tracking import MlflowClient

        from src.experiments.registry import verify_registry

        verify_registry(root, MlflowClient(tracking_uri="sqlite:///" + database.as_posix()))
        temporary = directory / ".publishing.db"
        temporary.unlink(missing_ok=True)
        backup_database(database, temporary)
        rewrite_locations(temporary, root, portable=True)
        # Идентифицированные записи с файлами обязаны оставаться доступными.
        with closing(sqlite3.connect(temporary)) as connection:
            directories = []
            for table, column in (("runs", "artifact_uri"), ("logged_models", "artifact_location")):
                for (location,) in connection.execute(f"SELECT {column} FROM {table}"):
                    if not (root / location).is_dir():
                        raise ValueError(f"Отсутствуют артефакты MLflow: {location}")
                    directories.append(location)
        files = {}
        for path in sorted((directory / "files").rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Снимок не допускает символические ссылки: {path}")
            if path.is_file():
                files[path.relative_to(directory).as_posix()] = sha256(path)
        files["tracking.db"] = sha256(temporary)
        # БД и список хешей проверяются вместе при восстановлении; частичный снимок отклоняется.
        temporary.replace(directory / "tracking.db")
        snapshot = {"format": 1, "files": files, "directories": sorted(set(directories))}
        write_json(directory / "snapshot.json", snapshot)
        write_json(directory / ".runtime/origin.json", {"published_sha256": files["tracking.db"]})
        return snapshot


def restore(root: Path) -> None:
    """Явное действие: локальная история архивируется, опубликованный снимок восстанавливается."""
    from datetime import UTC, datetime

    verify_snapshot(root)
    directory = root / "mlflow/.runtime"
    if directory.exists():
        archive = (
            root / ".local-archive" / ("mlflow-runtime-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f"))
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), archive)
    runtime_database(root)
