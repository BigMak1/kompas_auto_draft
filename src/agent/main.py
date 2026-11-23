"""Команды терминала: выбор чертежей, запуск агента, построение и оценка."""

import argparse
import asyncio
from pathlib import Path

from src.agent import agent
from src.config import ROOT, Config, load_config
from src.utils import Run, error_details, log, log_error, read_json, sha256


def select_drawings(drawing: str | None, config: Config) -> list[tuple[Path, Run]]:
    """Без аргумента берём весь набор из описи; PNG можно передать отдельно."""
    if drawing is not None and Path(drawing).suffix.lower() == ".png":
        source = Path(drawing)
        return [(source, Run(source.stem.upper(), config.name, ROOT))]
    items = read_json(ROOT / "data/manifest.json")["items"]
    names = [item["pair_id"] for item in items]
    if not names or len(names) != len(set(names)):
        raise ValueError("Опись должна содержать непустой список деталей без повторов")
    if drawing is not None:
        items = [item for item in items if item["pair_id"] == drawing.upper()]
        if not items:
            raise ValueError(f"Деталь {drawing} отсутствует в описи. Для своего чертежа укажите путь к PNG")
    selected = []
    for item in items:
        source = ROOT / "data" / item["input_png"]["path"]
        if sha256(source) != item["input_png"]["sha256"]:
            raise ValueError(f"SHA256 не совпал: {source.name}")
        selected.append((source, Run(item["pair_id"], config.name, ROOT)))
    return selected


async def run_drawings(selected: list[tuple[Path, Run]], config: Config) -> int:
    successful = 0
    for index, (source, run) in enumerate(selected, 1):
        log("НАБОР", f"{index}/{len(selected)} · {run.pair_id}")
        record = await agent.generate(source, run, config)
        successful += record["status"] == "success"
    failed = len(selected) - successful
    log("ИТОГ", f"Построено {successful}/{len(selected)}. Неудачных запусков: {failed}.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.agent",
        description="Baseline: один ответ LLM → рецепт → STL",
        epilog="Команды проекта: uv run poe help. Весь набор: uv run poe run.",
    )
    commands = parser.add_subparsers(dest="command")
    generate_parser = commands.add_parser("run", help="Построить STL для всего набора или одного PNG")
    generate_parser.add_argument(
        "drawing", nargs="?", help="KF0027 или путь к PNG; без аргумента — все детали из data/manifest.json"
    )
    build_parser = commands.add_parser("build", help="Построить STL из рецепта без LLM")
    build_parser.add_argument("part", help="Название детали для папки прогона")
    build_parser.add_argument("recipe", type=Path)
    evaluate_parser = commands.add_parser("evaluate", help="Посчитать две метрики отдельно от генерации")
    evaluate_parser.add_argument("selector", nargs="?")
    commands.add_parser("check", help="Проверить настройки и хеши данных без API")
    for subparser in (generate_parser, build_parser, evaluate_parser):
        subparser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    stage = "чтение конфигурации"
    try:
        config = load_config(args.config) if hasattr(args, "config") else load_config()
        if args.command == "check":
            stage = "проверка данных и настроек"
            from src.evaluate import check_data

            check_data(ROOT / "data")
            agent.make_agent(config.agent)
            log("ГОТОВО", "Конфиг и хеши данных проверены. Запросов к API нет.")
            return 0
        if args.command == "evaluate":
            stage = "оценка и сохранение метрик"
            from src.evaluate import evaluate_selected

            return evaluate_selected(args.selector, config.evaluation, run_name=config.name)
        if args.command == "run":
            stage = "проверка входных файлов"
            selected = select_drawings(args.drawing, config)
            stage = "подготовка запуска"
            return asyncio.run(run_drawings(selected, config))
        stage = "чтение сохранённого рецепта и подготовка построения"
        record = agent.replay(args.recipe, Run(args.part.upper(), config.name, ROOT), config)
        return 0 if record["status"] == "success" else 1
    except Exception as error:  # noqa: BLE001 — ошибка команды тоже должна остаться на диске.
        log_error(error_details(error, stage), ROOT / "artifacts/command-error.log")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
