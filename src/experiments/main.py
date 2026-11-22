"""Единый CLI проекта: каталог, проверка, запуск, офлайн-оценка, синхронизация MLflow и сборка STL."""

import argparse
import asyncio
from pathlib import Path

from filelock import FileLock

from src.cad.builder import build
from src.core.config import (
    ExperimentConfig,
    catalogue,
    default_study,
    load_experiment,
    resolve_experiment,
    resolve_prompts,
)
from src.core.logging import journal, log, log_error
from src.core.utils import ROOT, error_details, read_json, safe_name
from src.evaluation.dataset import check_manifest
from src.experiments.report import evaluate_study, observations, report_study
from src.experiments.runner import run_matrix, validate_config_pipelines
from src.experiments.store import (
    attempts,
    case_name,
    purge_study_files,
    read_record,
    study_path,
    verify_study,
)
from src.schemas.recipe import Recipe


def show_experiment(path: Path, config: ExperimentConfig) -> None:
    number = f"№{config.number:02d}" if config.number is not None else "Свой конфиг"
    log("ИССЛЕДОВАНИЕ", f"{number} · {config.title} ({config.name})")
    if config.description:
        log("ЦЕЛЬ", " ".join(config.description.split()))
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    log("КОНФИГ", f"{rel}/" if path.is_dir() else str(rel))
    for key, variant in config.variants.items():
        log("ВАРИАНТ", f"{key} [{variant.role}] · {variant.label}: {variant.description}")
    log(
        "МОДЕЛЬ",
        f"{config.agent.model}, reasoning={config.agent.reasoning_effort}, "
        f"concurrency={config.execution.concurrency}",
    )
    log(
        "БЮДЖЕТ",
        f"На маршрут: {config.agent.timeout_seconds} с, "
        f"{config.agent.max_output_tokens} выходных токенов (чтение ≤ {config.reading.max_output_tokens}).",
    )


def succeeded(directory: Path) -> bool:
    study = verify_study(directory)
    for job in study["schedule"]:
        found = attempts(directory / job["variant"], case_name(job["pair_id"], job["repeat"]))
        if not found or read_record(found[-1])["status"] != "success":
            return False
    return True


def _study_lock(root: Path, name: str) -> Path:
    lock = root / "logs/.locks" / f"{safe_name(name)}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    return lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Исследования: каталог, проверка, параллельный запуск, оценка, MLflow и сборка STL",
        epilog="Примеры: uv run poe exp list · uv run poe exp show 1 · uv run poe exp check 1",
    )
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("list", help="Список всех исследований в configs/experiments/")

    show = commands.add_parser("show", help="Показать параметры, варианты и бюджеты без API")
    show.add_argument("experiment", nargs="?", help="Номер или имя из каталога (по умолчанию 1)")
    show.add_argument("--config", type=Path, help="Путь к YAML-конфигу")

    check = commands.add_parser("check", help="Проверить YAML, промпты и манифест датасета без API")
    check.add_argument("experiment", nargs="?", help="Номер или имя из каталога (по умолчанию 1)")
    check.add_argument("--config", type=Path, help="Путь к YAML-конфигу")
    check.add_argument("--dev-only", action="store_true", help="Не требовать готовность val выборки")

    run = commands.add_parser("run", help="Запустить или продолжить матрицу эксперимента")
    run.add_argument("experiment", nargs="?", help="Номер исследования (по умолчанию 1)")
    run.add_argument("--config", type=Path, help="Свой YAML вместо номера исследования")
    run.add_argument("--split", choices=("dev", "val"), required=True)
    run.add_argument("--pair", action="append", type=str.upper, help="Конкретная деталь (можно несколько)")
    run.add_argument("--repeats", type=int, help="Переопределить число повторов")
    run.add_argument("--concurrency", type=int, help="Число параллельных задач")
    run.add_argument("--study", help="Своё имя серии")
    run.add_argument("--retry-infrastructure", action="store_true", help="Повторить только тех. сбои")
    run.add_argument("--no-eval", action="store_true", help="Пропустить авто-оценку и отчёт после прогона")
    run.add_argument("--no-tracking", action="store_true", help="Выполнить без записи в MLflow")

    for cmd_name, help_text in (
        ("evaluate", "Офлайн-расчёт метрик, отчёта и обновление MLflow"),
        ("report", "Пересобрать report.md и графики по сохранённым оценкам"),
        ("sync", "Восстановить/досинхронизировать серию в MLflow из локальных файлов"),
    ):
        sub = commands.add_parser(cmd_name, help=help_text)
        sub.add_argument("--study", required=True, help="Имя серии в runs/")
        if cmd_name == "evaluate":
            sub.add_argument("--no-tracking", action="store_true", help="Не обновлять MLflow")

    purge = commands.add_parser(
        "purge",
        help="Навсегда удалить серию из MLflow (runs, traces, артефакты) и с диска",
    )
    purge.add_argument("--study", required=True, help="Имя серии для удаления")
    purge.add_argument(
        "--mlflow-only",
        action="store_true",
        help="Удалить только из MLflow, оставив runs/ и results/ на диске",
    )

    bld = commands.add_parser("build", help="Детерминированная пересборка STL из recipe.json без LLM")
    bld.add_argument("part", help="Имя выходной подпапки/детали в runs/_manual/")
    bld.add_argument("recipe", type=Path, help="Путь к JSON-файлу рецепта")
    bld.add_argument("--config", type=Path, help="YAML-конфиг с секцией cad (по умолчанию 1)")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "list":
            for path, cfg in catalogue():
                loc = f"{path.name}/" if path.is_dir() else path.name
                log("КАТАЛОГ", f"{cfg.number:02d}. {cfg.title} ({cfg.name}) · {loc}")
                if cfg.description:
                    log("ЦЕЛЬ", " ".join(cfg.description.split()))
            log("СПРАВКА", "Подробнее: uv run poe exp show 1")
            return 0

        if args.command in {"show", "check", "run"}:
            path, config = resolve_experiment(args.experiment, getattr(args, "config", None))
            if args.command == "show":
                show_experiment(path, config)
                return 0
            if args.command == "check":
                with journal(ROOT, config.name, "check"):
                    show_experiment(path, config)
                    validate_config_pipelines(config)
                    config = resolve_prompts(ROOT, config)
                    check_manifest(
                        ROOT,
                        config.manifest,
                        validation_size=config.execution.validation_size,
                        require_val=not args.dev_only,
                    )
                    log(
                        "ГОТОВО",
                        "Конфиг, промпты, пайплайны и манифест датасета проверены; API не вызывался.",
                    )
                return 0

            repeats = args.repeats if args.repeats is not None else config.execution.repeats
            name = args.study or default_study(config, args.split, args.pair, repeats)
            directory = study_path(ROOT, name)
            with FileLock(_study_lock(ROOT, directory.name), timeout=0), journal(ROOT, name, "run"):
                show_experiment(path, config)
                asyncio.run(
                    run_matrix(
                        ROOT,
                        config,
                        name,
                        args.split,
                        args.pair,
                        repeats,
                        concurrency=args.concurrency,
                        retry_infrastructure=args.retry_infrastructure,
                        auto_evaluate=not args.no_eval,
                        tracking=not args.no_tracking,
                    )
                )
                return 0 if succeeded(directory) else 1

        if args.command == "build":
            cfg = load_experiment(args.config) if args.config else resolve_experiment(None, None)[1]
            recipe = Recipe.model_validate(read_json(args.recipe))
            out_dir = ROOT / "runs/_manual" / safe_name(args.part)
            info = build(recipe, out_dir, cfg.cad)
            log("ГОТОВО", f"STL собран: {out_dir / 'model.stl'} (объём {info['volume_mm3']:.2f} мм³)")
            return 0

        if args.command == "purge":
            from src.tracking.mlflow import purge_study_from_mlflow

            safe = safe_name(args.study)
            with journal(ROOT, "commands", "purge"):
                lock_file = _study_lock(ROOT, safe)
                had_lock = lock_file.exists()
                with FileLock(lock_file, timeout=0):
                    pass
                if args.mlflow_only and not had_lock:
                    lock_file.unlink(missing_ok=True)
                stats = purge_study_from_mlflow(ROOT, safe)
                removed = [] if args.mlflow_only else purge_study_files(ROOT, safe)
                if not any(stats.values()) and not removed:
                    log("ОЧИСТКА", f"Для серии `{safe}` не найдено данных в MLflow и на диске.")
                else:
                    dirs_note = f"; с диска: {', '.join(removed)}" if removed else ""
                    log(
                        "ОЧИСТКА",
                        f"Серия `{safe}` удалена: MLflow runs={stats['runs']}, "
                        f"traces={stats['traces']}, artifacts={stats['artifact_dirs']}{dirs_note}.",
                    )
            return 0

        directory = study_path(ROOT, args.study)
        with journal(ROOT, args.study, args.command):
            if args.command == "evaluate":
                evaluate_study(ROOT, args.study)
                summary = report_study(ROOT, args.study)
                if not args.no_tracking:
                    from src.tracking.mlflow import sync_study_results

                    study = verify_study(directory)
                    rows = observations(ROOT, directory, study)
                    sync_study_results(ROOT, directory, study, rows, summary, ROOT / "results" / args.study)
                return 0 if summary["complete"] else 1
            if args.command == "report":
                summary = report_study(ROOT, args.study)
                return 0 if summary["complete"] else 1
            if args.command == "sync":
                from src.tracking.mlflow import sync_study_results

                study = verify_study(directory)
                rows = observations(ROOT, directory, study)
                summary = report_study(ROOT, args.study)
                sync_study_results(ROOT, directory, study, rows, summary, ROOT / "results" / args.study)
                return 0
    except Exception as error:  # noqa: BLE001 — диагностируем любую ошибку команды на диск.
        with journal(ROOT, "commands", "errors"):
            log_error(error_details(error, "команда CLI"), ROOT / "logs/commands/experiment-error.log")
        return 1
    return 0
