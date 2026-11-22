"""Общие утилиты: файловый ввод-вывод, хеши, санитаризация секретов и проверка входов."""

import hashlib
import json
import os
import re
import traceback
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def safe_name(name: str) -> str:
    if not SAFE_NAME_PATTERN.fullmatch(name):
        raise ValueError("Идентификатор: латинские буквы, цифры, дефис или подчёркивание")
    return name


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def redact(value: Any) -> Any:
    """Убираем секреты и base64-изображения из журналов, диагностик и трасс."""
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
    """Текст, цепочка причин и стек без локальных переменных, запросов и заголовков."""
    chain = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        item: dict[str, Any] = {
            "type": type(current).__name__,
            "message": str(current) or "Без сообщения",
        }
        if getattr(current, "__notes__", None):
            item["notes"] = list(current.__notes__)
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


def require_single_sheet(item: dict) -> None:
    """Текущий протокол принимает только проверенные однолистовые чертежи."""
    count = item.get("sheet_count")
    if type(count) is not int or count != 1:
        raise ValueError(f"{item['pair_id']}: в манифесте должен быть подтверждён ровно один лист")


def require_orientation(item: dict) -> str:
    """В контекст модели разрешена только ручная подсказка, не вся карточка эталона."""
    orientation = item.get("orientation")
    if not isinstance(orientation, str) or not orientation.strip():
        raise ValueError(f"{item['pair_id']}: в манифесте должна быть задана ориентация детали")
    return orientation
