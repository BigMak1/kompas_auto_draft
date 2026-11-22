"""Стандартный logging: терминал, журналы команды/попыток и безопасная диагностика."""

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

from src.utils import redact

_logger = logging.getLogger(__name__)
_scope: ContextVar[str] = ContextVar("experiment_scope", default="")
_files: ContextVar[dict[Path, logging.FileHandler] | None] = ContextVar("experiment_logs", default=None)


class SafeFormatter(logging.Formatter):
    """Маскирование применяется и к сообщениям сторонних библиотек, и к traceback."""

    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        record.stage = getattr(record, "stage", "SDK")
        context = getattr(record, "scope", _scope.get())
        prefix = f"[{context}] " if context and not getattr(record, "diagnostic", False) else ""
        record.msg, record.args = prefix + record.getMessage(), ()
        return redact(super().format(record))


def normal(record: logging.LogRecord) -> bool:
    return not getattr(record, "diagnostic", False)


def console_handler() -> RichHandler:
    handler = RichHandler(console=Console(), show_time=False, show_level=False, show_path=False)
    handler.setFormatter(SafeFormatter("%(asctime)s  %(stage)-10s %(message)s", datefmt="%H:%M:%S"))
    handler.addFilter(normal)
    return handler


def file_handler(path: Path, *, mode: str = "a", diagnostic: bool = False) -> logging.FileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    handler.setFormatter(SafeFormatter("%(asctime)s  %(stage)-10s %(message)s"))
    handler.addFilter(lambda record: bool(getattr(record, "diagnostic", False)) == diagnostic)
    return handler


@contextmanager
def journal(root: Path, study: str, command: str) -> Iterator[Path]:
    """Обработчики живут одну команду; старые журналы дописываются."""
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
    token = _files.set({})
    try:
        log("ЖУРНАЛ", f"{command} · {path.relative_to(root)}")
        yield path
    except BaseException as error:
        log("ПРЕРВАНО", f"{type(error).__name__}: {error}")
        raise
    finally:
        for handler in [*handlers, sdk_handler, *(_files.get() or {}).values()]:
            handler.close()
        for logger, old_handlers, level, propagate in previous:
            logger.handlers, logger.level, logger.propagate = old_handlers, level, propagate
        _files.reset(token)


@contextmanager
def scope(label: str) -> Iterator[None]:
    token = _scope.set(label)
    try:
        yield
    finally:
        _scope.reset(token)


def log(stage: str, message: str, path: Path | None = None) -> None:
    files = _files.get()
    if files is not None and path is not None and path not in files:
        handler = file_handler(path)
        handler.addFilter(lambda record: getattr(record, "attempt_log", None) == path)
        files[path] = handler
        _logger.addHandler(handler)
    level = logging.ERROR if stage == "ОШИБКА" else logging.WARNING if stage == "ВНИМАНИЕ" else logging.INFO
    extra = {"stage": stage, "scope": _scope.get(), "attempt_log": path}
    if files is not None:
        _logger.log(level, message, extra=extra)
        return
    # Отчёт и офлайн-тесты могут вызывать функцию без CLI-контекста journal.
    record = _logger.makeRecord(_logger.name, level, "", 0, message, (), None, extra=extra)
    handlers = [console_handler(), *([file_handler(path)] if path is not None else [])]
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
            _logger.name, logging.ERROR, "", 0, body, (), None, extra={"stage": "ОШИБКА", "diagnostic": True}
        )
        try:
            handler.handle(record)
            if _files.get() is not None:
                _logger.handle(record)
        finally:
            handler.close()
        log("ПОДРОБНО", str(path), logfile)
    except OSError as error:
        log("ВНИМАНИЕ", f"Не удалось записать диагностику: {error}", logfile)


async def waiting[T](
    operation: Awaitable[T], label: str, timeout: float, interval: float, logfile: Path | None = None
) -> T:
    """Реальное время ожидания; лимит контролирует сам запрос или построитель."""
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
                f"{label}: прошло {elapsed:.0f} с, до лимита {max(0, timeout - elapsed):.0f} с.",
                logfile,
            )
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def log_response(response: dict, limit: int, logfile: Path) -> None:
    usage = response.get("usage")
    seconds = response.get("seconds", 0)
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
