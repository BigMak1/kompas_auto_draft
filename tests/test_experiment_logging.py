"""Ожидание и журналы проверяются без запросов к API."""

import asyncio
import logging
from pathlib import Path

import pytest

from src.experiments.logging import journal, log, log_error, scope, waiting
from src.utils import error_details


def test_waiting_logs_progress_and_does_not_repeat_operation(tmp_path: Path) -> None:
    calls = []

    async def operation() -> str:
        calls.append("one")
        await asyncio.sleep(0.035)
        return "done"

    with journal(tmp_path, "unit", "run"), scope("KF0012 · Только паспорт"):
        assert asyncio.run(waiting(operation(), "ответ LLM", 10, 0.01)) == "done"
        log("ГОТОВО", "Ответ получен")
    path = tmp_path / "logs/unit/run.log"
    original = path.read_text()
    assert calls == ["one"] and "ОЖИДАНИЕ" in original and "KF0012 · Только паспорт" in original
    assert "до лимита" in original and "\x1b" not in original
    with journal(tmp_path, "unit", "run"):
        log("ПРОПУСК", "Сохранённый результат")
    assert path.read_text().startswith(original)


def test_waiting_cancellation_stops_background_operation() -> None:
    stopped = []

    async def operation() -> None:
        try:
            await asyncio.sleep(100)
        finally:
            stopped.append(True)

    async def cancel() -> None:
        task = asyncio.create_task(waiting(operation(), "LLM", 10, 1))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert stopped == [True]


def test_diagnostics_are_redacted_and_sdk_logger_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-private-fixture")
    logger = logging.getLogger("openai.agents")
    previous = logger.handlers[:], logger.propagate, logger.level
    with journal(tmp_path, "unit", "run"):
        logger.error("SDK message sk-proj-private-fixture data:image/png;base64,AAAA")
        try:
            raise ValueError("provider error sk-proj-private-fixture")
        except ValueError as error:
            details = error_details(error, "чтение паспорта")
            logger.exception("SDK traceback")
        details["summary"] = "Ответ обрезан по лимиту 8192 токенов."
        log_error(details, tmp_path / "artifacts/error.log")
    assert (logger.handlers, logger.propagate, logger.level) == previous
    terminal = capsys.readouterr().out
    assert "Ответ обрезан" in terminal and "SDK message" not in terminal
    sdk = (tmp_path / "logs/unit/sdk.log").read_text()
    assert "SDK message" in sdk and "[PNG СКРЫТ]" in sdk
    assert "Traceback" in sdk
    for path in tmp_path.rglob("*.log"):
        text = path.read_text()
        assert "sk-proj-private-fixture" not in text and "base64,AAAA" not in text
    assert "Traceback" in (tmp_path / "logs/unit/errors.log").read_text()
