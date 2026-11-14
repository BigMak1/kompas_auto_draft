"""Общие файлы прогонов и короткие сообщения; без зависимостей от агента и CAD."""

import hashlib
import json
import os
import re
import shutil
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.text import Text

from src.config import ROOT

console = Console(highlight=False)


def redact(value: Any) -> Any:
    """Убираем секреты и изображения из диагностических сообщений и ответов API."""
    if isinstance(value, dict):
        return {
            key: "[СКРЫТО]"
            if re.search(r"(?i)(?:^|[_-])(?:api[_-]key|token|secret|password)$|^authorization$", key)
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if not isinstance(value, str):
        return value
    for name, secret in os.environ.items():
        if secret and name.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
            value = value.replace(secret, "[СКРЫТО]")
    value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[СКРЫТО]", value)
    value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [СКРЫТО]", value)
    value = re.sub(
        r"""(?i)(\b(?:authorization|api[_-]?key|access_token|refresh_token|password|secret)\b["']?\s*[:=]\s*)"""
        r"""(?:"[^"]*"|'[^']*'|[^\s,;}\]]+)""",
        r"\1[СКРЫТО]",
        value,
    )
    return re.sub(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=\r\n]+", "[PNG СКРЫТ]", value)


def error_details(error: BaseException, stage: str) -> dict:
    """Текст, причины и стек без локальных переменных, запросов и HTTP-заголовков."""
    chain = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        item = {"type": type(current).__name__, "message": str(current) or "Без сообщения"}
        if getattr(current, "__notes__", None):
            item["notes"] = current.__notes__
        for field in ("status_code", "request_id", "code", "param"):
            value = getattr(current, field, None)
            if isinstance(value, (str, int)):
                item[field] = value
        if isinstance(current, ValidationError):
            item["validation"] = current.errors(include_input=False, include_context=False, include_url=False)
        chain.append(item)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return redact(
        {
            "stage": stage,
            **chain[0],
            "causes": chain[1:],
            "traceback": "".join(traceback.format_exception(error)),
        }
    )


def log_error(details: dict, path: Path, logfile: Path | None = None) -> None:
    """Короткая причина в основном логе; полная диагностика в отдельном файле."""
    details = redact(details)
    message = f"Этап: {details['stage']}. {details['type']}: {details['message']}"
    for field, label in (("status_code", "HTTP"), ("code", "код"), ("request_id", "ID запроса")):
        if field in details:
            message += f"; {label}: {details[field]}"
    if details.get("causes"):
        cause = details["causes"][-1]
        message += f"; причина: {cause['type']}: {cause['message']}"
    short_message = message
    validation = details.get("validation") or next(
        (cause["validation"] for cause in details.get("causes", []) if cause.get("validation")), None
    )
    if validation:
        fields = "; ".join(
            f"{'.'.join(map(str, item['loc'])) or 'JSON'}: {item['msg']}" for item in validation
        )
        short_message = f"Этап: {details['stage']}. {details['type']}. Ошибки проверки: {fields}"
    if details.get("notes"):
        short_message += "; " + "; ".join(details["notes"])
    short = " ".join(short_message.split())
    log("ОШИБКА", short if len(short) <= 800 else short[:797] + "…", logfile)
    try:
        body = f"{datetime.now().astimezone().isoformat()}\n{message}\n\n{details['traceback']}"
        atomic(path, body.encode("utf-8"))
    except OSError as error:
        log("ОШИБКА", f"Не удалось сохранить диагностику: {redact(str(error))}", logfile)
    else:
        display = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        log("ДИАГНОСТИКА", str(display), logfile)


def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(stage: str, message: str, path: Path | None = None) -> None:
    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
    line = f"{stamp}  {stage:<10} {redact(message)}"
    color = {"ГОТОВО": "green", "ОШИБКА": "red", "ВНИМАНИЕ": "yellow"}.get(stage, "cyan")
    console.print(Text(line, style=color))
    if path:
        try:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError as error:
            console.print(Text(f"Не удалось записать {path}: {redact(str(error))}", style="red"))


@dataclass(frozen=True)
class Run:
    pair_id: str
    run_name: str = "baseline"
    root: Path = ROOT

    def __post_init__(self) -> None:
        for value in (self.pair_id, self.run_name):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
                raise ValueError("Имя детали и прогона: латинские буквы, цифры, дефис или подчёркивание")

    @property
    def name(self) -> str:
        return f"{self.run_name}/{self.pair_id}"

    @property
    def directory(self) -> Path:
        return self.root / "run" / self.name

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts" / self.name

    @property
    def logfile(self) -> Path:
        return self.directory / "run.log"

    @property
    def report(self) -> Path:
        return self.root / "results" / self.run_name / "metrics.md"

    def create(self) -> None:
        """Новый запуск заменяет результат этой детали и сбрасывает старую оценку."""
        for path in (self.directory, self.artifacts):
            if path.is_symlink():
                raise ValueError("Папка прогона не должна быть символической ссылкой")
        for path in (self.directory, self.artifacts):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)
        self.report.unlink(missing_ok=True)

    def log(self, stage: str, message: str) -> None:
        log(stage, message, self.logfile)


def select_runs(selector: str | None, root: Path = ROOT, run_name: str = "baseline") -> list[Run]:
    if selector is None:
        items = read_json(root / "data/manifest.json")["items"]
        return [Run(item["pair_id"], run_name, root) for item in items]
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", selector):
        return [Run(selector.upper(), run_name, root)]
    candidate = Path(selector)
    if not candidate.is_absolute():
        candidate = root / candidate
    relative = candidate.resolve().relative_to((root / "run" / run_name).resolve())
    if len(relative.parts) != 1:
        raise ValueError("Укажите деталь или папку run/<название>/<деталь>")
    return [Run(relative.name, run_name, root)]
