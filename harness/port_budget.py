"""Запас локальных TCP-портов при частых вызовах KsAPI v25."""

import os
import re
import subprocess
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from time import monotonic, sleep
from typing import Any

CALLS_PER_SAMPLE = 512
MAX_PORT_WAIT = 250.0


def windows_output(executable: str, *args: str) -> bytes:
    path = Path(os.environ["SystemRoot"]) / "System32" / executable
    return subprocess.check_output(
        [str(path), *args],
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def dynamic_range(output: bytes) -> range:
    start, count = (int(value) for value in re.findall(rb":\s*(\d+)\s*$", output, re.M))
    return range(start, start + count)


def occupied_ports(output: bytes, ports: range) -> int:
    occupied = set()
    for line in output.splitlines():
        columns = line.split()
        if not columns or columns[0] != b"TCP" or len(columns) < 4:
            continue
        address, separator, value = columns[1].rpartition(b":")
        if separator and b"." in address and value.isdigit() and (port := int(value)) in ports:
            occupied.add(port)
    return len(occupied)


class PortBudget:
    def __init__(self, ports: range, sample: Callable[[], int]) -> None:
        self.sample = sample
        self.limit = len(ports) * 3 // 4
        self.batch = min(CALLS_PER_SAMPLE, max(1, len(ports) // 16))
        self.remaining = 0

    def before_call(self) -> None:
        if self.remaining:
            self.remaining -= 1
            return
        deadline = monotonic() + MAX_PORT_WAIT
        occupied = self.sample()
        if occupied >= self.limit:
            print(f"KsAPI TCP port budget: {occupied} occupied, waiting below {self.limit}", flush=True)
        while occupied >= self.limit:
            if monotonic() >= deadline:
                raise TimeoutError(f"KsAPI TCP port budget exhausted: {occupied} occupied")
            sleep(3)
            occupied = self.sample()
        self.remaining = self.batch - 1


class PacedLibrary:
    """Проверяет бюджет между пакетами вызовов, сохраняя функции загруженной DLL."""

    def __init__(self, library: Any, budget: PortBudget) -> None:
        self.library = library
        self.budget = budget

    def __getattr__(self, name: str) -> Any:
        native = getattr(self.library, name)
        if not callable(native):
            return native

        @wraps(native)
        def paced(*args: Any, **kwargs: Any) -> Any:
            self.budget.before_call()
            return native(*args, **kwargs)

        setattr(self, name, paced)
        return paced


def windows_port_budget() -> PortBudget:
    ports = dynamic_range(windows_output("netsh.exe", "interface", "ipv4", "show", "dynamicport", "tcp"))
    return PortBudget(
        ports, lambda: occupied_ports(windows_output("netstat.exe", "-q", "-n", "-p", "tcp"), ports)
    )
