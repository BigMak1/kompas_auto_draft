"""CLI: uv run poe eval-geometry <папка с evaluation.json>."""

import argparse
from pathlib import Path

from eval.geometry.backend import settings_from_path
from eval.geometry.io import read_json
from eval.geometry.pipeline import execute, load_evaluation
from eval.geometry.progress import progress_display
from eval.geometry.readiness import inspect_inputs
from eval.geometry.terminal import add_output_arguments, cache_message, duration, print_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--offline", action="store_true", help="Не обращаться к КОМПАС")
    parser.add_argument("--backend", type=Path, help="JSON настроек Windows-вычислителя")
    parser.add_argument("--prepare", action="store_true", help="Только подготовить сетки и Facts")
    parser.add_argument("--check-inputs", action="store_true", help="Проверить входы без вычислений и KsAPI")
    add_output_arguments(parser)
    args = parser.parse_args()
    try:
        if args.check_inputs:
            readiness = inspect_inputs(args.run.resolve(), load_evaluation(args.run.resolve()))
            for item in readiness["issues"]:
                print(f"{item['case_id']} / {item['role']}: {item['message']}")
            print("Входы поверхности готовы" if readiness["ready_for_surface"] else "Входы неполные")
            return 0 if readiness["ready_for_surface"] else 2
        backend_path = args.backend
        if backend_path is None and (args.run / "evaluation/backend.json").is_file():
            backend_path = args.run / "evaluation/backend.json"
        with progress_display(quiet=args.quiet, no_color=args.no_color) as progress:
            output, preparation = execute(
                args.run,
                offline=args.offline,
                backend=None if args.offline else settings_from_path(backend_path),
                prepare_only=args.prepare,
                progress=progress,
            )
    except (ValueError, OSError) as error:
        parser.exit(2, f"Ошибка входа оценки: {error}\n")
    if not args.quiet:
        if preparation["backend"] == "not_configured":
            print("Вычислитель КОМПАС не настроен; считаем доступные показатели по файлам.")
        if preparation.get("backend_error"):
            print(f"КОМПАС недоступен: {preparation['backend_error']}")
        for item in preparation["after"]["issues"]:
            print(f"{item['case_id']} / {item['role']}: {item['message']}")
        for action in preparation["actions"]:
            if "error" in action:
                print(f"Ошибка подготовки: {action['error']}")
            if message := cache_message(action):
                print(message)
    if args.prepare:
        if not args.quiet:
            print(f"Общее время прогона: {duration(preparation['elapsed_seconds'])}")
        print(output)
        return 0 if preparation["after"]["ready_for_surface"] else 2
    report = read_json(output / "metrics.json")
    print_summary(
        report, output, all_metrics=args.all_metrics, no_color=args.no_color, quiet=args.quiet,
        elapsed_seconds=preparation["elapsed_seconds"],
    )
    # Неполный IoU не должен выглядеть успешным завершением batch-оценки.
    return 0 if report["summary"]["rankable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
