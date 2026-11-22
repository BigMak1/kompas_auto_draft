"""Каталог исследований, запуск по номеру и работа с сохранёнными сериями."""

import argparse
import asyncio
from pathlib import Path

from src.config import ROOT
from src.experiments.config import (
    DESCRIPTIONS,
    LABELS,
    ExperimentConfig,
    catalogue,
    check_manifest,
    default_study,
    resolve_experiment,
)
from src.experiments.logging import journal, log, log_error
from src.experiments.storage import study_lock
from src.experiments.store import attempts, case_name, read_record, study_path, verify_study
from src.utils import error_details


def show_experiment(path: Path, config: ExperimentConfig) -> None:
    number = f"№{config.number}" if config.number is not None else "Свой конфиг"
    log("ИССЛЕДОВАНИЕ", f"{number} · {config.title}")
    if config.description:
        log("ЦЕЛЬ", config.description)
    log("КОНФИГ", str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))
    for variant in config.variants:
        log("МАРШРУТ", f"{LABELS[variant]}: {DESCRIPTIONS[variant]}")
    log("МОДЕЛЬ", f"{config.agent.model}, рассуждения {config.agent.reasoning_effort}")
    log(
        "БЮДЖЕТ",
        f"На маршрут: {config.agent.timeout_seconds} с, "
        f"{config.agent.max_output_tokens} выходных токенов; чтение ≤{config.reading.max_output_tokens}.",
    )


def succeeded(directory: Path) -> bool:
    study = verify_study(directory)
    for job in study["schedule"]:
        found = attempts(directory / job["variant"], case_name(job["pair_id"], job["repeat"]))
        if not found or read_record(found[-1])["status"] != "success":
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Эксперименты: каталог, запуск по номеру, оценка и отчёт",
        epilog="Каталог: uv run poe experiment list. Описание: uv run poe experiment show 1.",
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("list", help="Номера, названия и цели исследований")
    show = commands.add_parser("show", help="Маршруты и условия одного исследования без API")
    show.add_argument("experiment", help="Номер из каталога, например 1")
    run = commands.add_parser("run", help="Выполнить или продолжить серию")
    run.add_argument("experiment", nargs="?", help="Номер исследования; по умолчанию 1")
    run.add_argument("--config", type=Path, help="Свой YAML вместо номера исследования")
    run.add_argument("--split", choices=("dev", "val"), required=True)
    run.add_argument("--pair", action="append", type=str.upper)
    run.add_argument("--repeats", type=int)
    run.add_argument("--study", help="Своё имя серии; иначе оно составляется из конфига и выборки")
    run.add_argument(
        "--retry-infrastructure", action="store_true", help="Отдельные попытки для технических ошибок"
    )
    for command in ("evaluate", "report"):
        sub = commands.add_parser(command)
        sub.add_argument("--study", required=True)
    track = commands.add_parser("track", help="Перенести сохранённые результаты в MLflow без API")
    selection = track.add_mutually_exclusive_group(required=True)
    selection.add_argument("--study")
    selection.add_argument("--all", action="store_true", help="Все сохранённые серии")
    check = commands.add_parser("check", help="Проверить конфиг и данные без API")
    check.add_argument("experiment", nargs="?")
    check.add_argument("--config", type=Path)
    check.add_argument("--dev-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "list":
            for path, config in catalogue():
                log("КАТАЛОГ", f"{config.number}. {config.title} · {path.name}")
                log("ЦЕЛЬ", config.description)
            log(
                "СПРАВКА",
                "Условия: uv run poe experiment show 1. "
                "Запуск: uv run poe experiment run 1 --split dev --pair KF0012 --repeats 1",
            )
            return 0
        if args.command in {"show", "check", "run"}:
            path, config = resolve_experiment(args.experiment, getattr(args, "config", None))
            if args.command == "show":
                show_experiment(path, config)
                return 0
            if args.command == "check":
                with journal(ROOT, config.name, "check"):
                    show_experiment(path, config)
                    check_manifest(ROOT, config, require_val=not args.dev_only)
                    log("ГОТОВО", "Конфиг и манифест датасета проверены; запросов к API нет.")
                return 0
            from filelock import FileLock

            from src.experiments.runner import run_matrix

            config_path = path
            repeats = args.repeats if args.repeats is not None else config.repeats
            name = args.study or default_study(config, args.split, args.pair, repeats)
            directory = study_path(ROOT, name)
            lock = study_lock(ROOT, directory.name)
            lock.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(lock, timeout=0), journal(ROOT, name, "run"):
                show_experiment(config_path, config)
                asyncio.run(
                    run_matrix(
                        ROOT, config, name, args.split, args.pair, repeats, retry=args.retry_infrastructure
                    )
                )
                return 0 if succeeded(directory) else 1
        if args.command == "track" and args.all:
            from src.experiments.tracking import sync_all

            with journal(ROOT, "tracking", "sync"):
                return 0 if sync_all(ROOT) else 1
        directory = study_path(ROOT, args.study)
        with journal(ROOT, args.study, args.command):
            if args.command == "track":
                from filelock import FileLock

                from src.experiments.tracking import sync_study

                with FileLock(study_lock(ROOT, args.study), timeout=0):
                    sync_study(directory, ROOT, restore_deleted=True)
                log("ГОТОВО", "Сохранённая серия перенесена в локальный MLflow.")
                return 0
            from src.experiments.report import evaluate_study, report_study

            if args.command == "evaluate":
                evaluate_study(ROOT, args.study)
            summary = report_study(ROOT, args.study)
            return 0 if summary["complete"] else 1
    except Exception as error:  # noqa: BLE001 — сохраняем ошибку независимо от этапа.
        with journal(ROOT, "commands", "errors"):
            log_error(error_details(error, "эксперимент"), ROOT / "logs/commands/experiment-error.log")
        return 1
