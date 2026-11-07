"""Доверенный бюджет: общий срок и число вызовов build, включая ошибочные рецепты."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from harness.native_worker import atomic


def limits(run: Path) -> dict:
    path = run / "manifest.json"
    return json.loads(path.read_text()).get("limits", {}) if path.exists() else {}


def begin(run: Path) -> None:
    policy = limits(run)
    if not policy:
        return
    if not 1 <= policy["max_builds"] <= 30 or policy["max_seconds"] <= 0:
        raise ValueError("Invalid run limits")
    state = {
        "deadline_monotonic": time.monotonic() + policy["max_seconds"],
        "started_at": datetime.now(UTC).isoformat(),
        "build_calls": 0,
    }
    atomic(run / "control/budget.json", json.dumps(state, indent=2).encode())


def check(run: Path) -> dict | None:
    if not limits(run):
        return None
    state = json.loads((run / "control/budget.json").read_text())
    if time.monotonic() >= state["deadline_monotonic"]:
        raise TimeoutError("Generation budget expired; no further build or finish")
    return state


def reserve_build(run: Path) -> None:
    state = check(run)
    if state is None:
        return
    if state["build_calls"] >= limits(run)["max_builds"]:
        raise RuntimeError("Build call budget exhausted; select an existing successful attempt")
    state["build_calls"] += 1
    atomic(run / "control/budget.json", json.dumps(state, indent=2).encode())


def quarantine(run: Path, reason: str) -> None:
    if limits(run):
        marker = Path(__file__).resolve().parents[1] / "runs/hw01-quarantine.json"
        atomic(marker, json.dumps({"run": run.name, "reason": reason}, indent=2).encode())
