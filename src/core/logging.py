"""Стандартный logging: консоль, командный журнал и изолированные логи параллельных попыток."""

import asyncio
import copy
import logging
import time
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.core.utils import redact

_logger = logging.getLogger("kompas_auto_draft")
_scope: ContextVar[str] = ContextVar("experiment_scope", default="")
_attempt_log: ContextVar[Path | None] = ContextVar("attempt_log", default=None)
_active_registry: dict[Path, logging.FileHandler] | None = None


class SafeFormatter(logging.Formatter):
    """Применяет маскирование секретов и base64 ко всем сообщениям, включая сторонние SDK."""

    def format(self, record: logging.LogRecord) -> str:
        copied = copy.copy(record)
        copied.stage = getattr(copied, "stage", "SDK")
        context = getattr(copied, "scope", _scope.get())
        prefix = f"[{context}] " if context and not getattr(copied, "diagnostic", False) else ""
        copied.msg, copied.args = prefix + copied.getMessage(), ()
        return redact(super().format(copied))


def _is_normal(record: logging.LogRecord) -> bool:
    return not getattr(record, "diagnostic", False)


def console_handler() -> RichHandler:
    handler = RichHandler(console=Console(), show_time=False, show_level=False, show_path=False)
    handler.setFormatter(SafeFormatter("%(asctime)s  %(stage)-10s %(message)s", datefmt="%H:%M:%S"))
    handler.addFilter(_is_normal)
    return handler


def file_handler(path: Path, *, mode: str = "a", diagnostic: bool = False) -> logging.FileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    handler.setFormatter(SafeFormatter("%(asctime)s  %(stage)-10s %(message)s"))
    handler.addFilter(lambda record: bool(getattr(record, "diagnostic", False)) == diagnostic)
    return handler


@contextmanager
def journal(root: Path, study: str, command: str) -> Iterator[Path]:
    """Журнал одной CLI-команды: консоль + общий файл + errors.log + sdk.log."""
    global _active_registry
    path = root / "logs" / study / f"{command}.log"
    handlers = [
        console_handler(),
        file_handler(path),
        file_handler(path.parent / "errors.log", diagnostic=True),
    ]
    sdk_handler = file_handler(path.parent / "sdk.log")
    libraries = [logging.getLogger(name) for name in ("openai", "openai.agents", "mlflow")]
    previous = [
        (_logger, _logger.handlers[:], _logger.level, _logger.propagate),
        *((logger, logger.handlers[:], logger.level, logger.propagate) for logger in libraries),
    ]
    _logger.handlers, _logger.level, _logger.propagate = handlers[:], logging.INFO, False
    for logger in libraries:
        logger.handlers, logger.level, logger.propagate = [sdk_handler], logging.WARNING, False
    prev_registry = _active_registry
    _active_registry = {}
    try:
        display = path.relative_to(root) if path.is_relative_to(root) else path
        log("ЖУРНАЛ", f"{command} · {display}")
        yield path
    except BaseException as error:
        log("ПРЕРВАНО", f"{type(error).__name__}: {error}")
        raise
    finally:
        for handler in [*handlers, sdk_handler, *(_active_registry or {}).values()]:
            handler.close()
        for logger, old_handlers, level, propagate in previous:
            logger.handlers, logger.level, logger.propagate = old_handlers, level, propagate
        _active_registry = prev_registry


@contextmanager
def scope(label: str, logfile: Path | None = None) -> Iterator[None]:
    """Задаёт префикс [label] и опционально привязывает лог-файл текущей async-задачи."""
    scope_token = _scope.set(label)
    log_token = _attempt_log.set(logfile) if logfile is not None else None
    try:
        yield
    finally:
        if log_token is not None:
            _attempt_log.reset(log_token)
        _scope.reset(scope_token)
        if logfile is not None and _active_registry is not None and logfile in _active_registry:
            handler = _active_registry.pop(logfile)
            if handler in _logger.handlers:
                _logger.removeHandler(handler)
            handler.close()


def _ensure_attempt_handler(path: Path) -> None:
    if _active_registry is None or path in _active_registry:
        return
    handler = file_handler(path)
    handler.addFilter(lambda record, target=path: getattr(record, "attempt_log", None) == target)
    _active_registry[path] = handler
    _logger.addHandler(handler)


def log(stage: str, message: str, path: Path | None = None) -> None:
    target = path if path is not None else _attempt_log.get()
    if _active_registry is not None and target is not None:
        _ensure_attempt_handler(target)
    level = logging.ERROR if stage == "ОШИБКА" else logging.WARNING if stage == "ВНИМАНИЕ" else logging.INFO
    extra = {"stage": stage, "scope": _scope.get(), "attempt_log": target}
    if _active_registry is not None:
        _logger.log(level, message, extra=extra)
        return
    record = _logger.makeRecord(_logger.name, level, "", 0, message, (), None, extra=extra)
    handlers = [console_handler(), *([file_handler(target)] if target is not None else [])]
    for handler in handlers:
        try:
            handler.handle(record)
        finally:
            handler.close()


def log_error(details: dict, path: Path, logfile: Path | None = None) -> None:
    details = redact(details)
    message = details.get("summary") or f"{details['type']}: {details['message']}"
    if details.get("request_id"):
        message += f"; ID запроса: {details['request_id']}"
    log("ОШИБКА", f"{details['stage']}. {message}", logfile)
    body = (
        f"{_scope.get()}\nЭтап: {details['stage']}\n{details['type']}: {details['message']}\n"
        + "\n".join(details.get("notes", []))
        + "\n\n"
        + details.get("traceback", "")
    )
    try:
        handler = file_handler(path, mode="w", diagnostic=True)
        record = _logger.makeRecord(
            _logger.name,
            logging.ERROR,
            "",
            0,
            body,
            (),
            None,
            extra={"stage": "ОШИБКА", "diagnostic": True},
        )
        try:
            handler.handle(record)
            if _active_registry is not None:
                _logger.handle(record)
        finally:
            handler.close()
        log("ПОДРОБНО", str(path), logfile)
    except OSError as error:
        log("ВНИМАНИЕ", f"Не удалось записать диагностику: {error}", logfile)


async def waiting[T](
    operation: Awaitable[T], label: str, timeout: float, interval: float, logfile: Path | None = None
) -> T:
    """Ожидание операции с периодическим прогрессом в лог без повторного запуска."""
    started = time.monotonic()
    task = asyncio.ensure_future(operation)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                return await task
            elapsed = time.monotonic() - started
            log(
                "ОЖИДАНИЕ",
                f"{label}: прошло {elapsed:.0f} с, до лимита {max(0.0, timeout - elapsed):.0f} с.",
                logfile,
            )
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def log_response(response: dict, limit: int, logfile: Path | None = None) -> None:
    usage = response.get("usage")
    seconds = response.get("seconds", 0.0)
    if usage is None:
        log("ОТВЕТ", f"Этап завершён за {seconds:.1f} с; расход токенов API не вернул.", logfile)
        return
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    detail = f", рассуждения {reasoning}" if reasoning is not None else ""
    log(
        "ОТВЕТ",
        f"{seconds:.1f} с · вход {usage['input_tokens']} · "
        f"выход {usage['output_tokens']}/{limit} токенов{detail}.",
        logfile,
    )
