"""Текущий этап оценки: индикатор в терминале и периодические строки в логах."""

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event, Lock, Thread

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

ProgressCallback = Callable[[str], None]


def no_progress(message: str) -> None:
    """Вызовы из Python и тестов по умолчанию не печатают прогресс."""


def prefix_progress(progress: ProgressCallback, prefix: str) -> ProgressCallback:
    return lambda message: progress(f"{prefix} · {message}")


@contextmanager
def progress_display(*, quiet: bool = False, no_color: bool = False) -> Iterator[ProgressCallback]:
    if quiet:
        yield no_progress
        return
    console = Console(stderr=True, markup=False, highlight=False)
    if no_color or "NO_COLOR" in os.environ:
        console = Console(stderr=True, color_system=None, markup=False, highlight=False)
    if console.is_terminal and os.environ.get("TERM") != "dumb":
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("{task.description}", markup=False),
            TimeElapsedColumn(),
            console=console,
            transient=True,
            refresh_per_second=4,
        ) as display:
            task = display.add_task("Подготовка оценки", total=None)
            yield lambda message: display.update(task, description=message)
        return

    # В файле/CI нет управляющих последовательностей; долгий этап виден каждые 15 с.
    started = time.monotonic()
    stopped, lock = Event(), Lock()
    current = "Подготовка оценки"

    def write() -> None:
        elapsed = int(time.monotonic() - started)
        console.print(Text(f"[{elapsed // 60:02d}:{elapsed % 60:02d}] {current}"), soft_wrap=True)

    def update(message: str) -> None:
        nonlocal current
        with lock:
            current = message
            write()

    def heartbeat() -> None:
        while not stopped.wait(15):
            with lock:
                write()

    update(current)
    thread = Thread(target=heartbeat, name="eval-progress", daemon=True)
    thread.start()
    try:
        yield update
    finally:
        stopped.set()
        thread.join()
