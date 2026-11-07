"""Краткое представление готового отчёта; метрики здесь не пересчитываются."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

LABELS = {
    "solid_iou": "Объёмный IoU ↑",
    "f_score": "F-score ↑",
    "point_to_point.chamfer_l2_mm2": "Chamfer L2 (точки), мм² ↓",
    "siou_cadbench_formula_p2p": "SIoU CADBench (точки) ↑",
    "point_to_surface.hausdorff_mm": "Hausdorff (поверхность), мм ↓",
    "point_to_surface.hausdorff95_mm": "HD95 (поверхность), мм ↓",
    "edge_chamfer_brep_mm2": "ECD (B-rep), мм² ↓",
    "edge_chamfer_mesh_proxy_mm2": "ECD (рёбра сетки, proxy), мм² ↓",
    "normal_consistency": "Normal Consistency ↑",
    "dimension_absolute_error_mm_mean": "Ошибка размеров, мм ↓",
    "specification_compliance": "Выполнение требований ↑",
}
BRIEF_SECONDARY = tuple(key for key in LABELS if key not in ("solid_iou", "f_score"))


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--all-metrics", action="store_true", help="Вывести все вспомогательные метрики")
    parser.add_argument("--no-color", action="store_true", help="Отключить цвета в терминале")
    parser.add_argument("--quiet", action="store_true", help="Вывести только путь к результатам")


def number(value: float | None) -> Text:
    return Text(f"{value:.5g}") if value is not None else Text("—", style="yellow")


def duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours} ч {minutes:02d} мин {secs:02d} с" if hours else f"{minutes} мин {secs:02d} с"


def cache_message(action: dict[str, Any]) -> str | None:
    if not action.get("warning"):
        return None
    if action.get("status") == "restored":
        return "Кэш пересечений восстановлен: КОМПАС пересчитал недоступные результаты."
    if action.get("status") == "not_needed":
        return "Старый файл пересечений недоступен; для этого запуска он не потребовался."
    return "Файл пересечений недоступен; без восстановления нужных ответов КОМПАС IoU остаётся неполным."


def table(title: str, metrics: dict[str, Any], keys: list[str], total: int) -> Table:
    result = Table(
        title=title,
        title_style="bold cyan",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        header_style="dim",
        highlight=False,
    )
    result.add_column("Метрика")
    result.add_column("Среднее", justify="right", no_wrap=True)
    result.add_column("Медиана", justify="right", no_wrap=True)
    result.add_column("Учтено", justify="right", no_wrap=True)
    for key in keys:
        item = metrics.get(key, {})
        count = item.get("measured_count", 0)
        denominator = item.get("total_count", total)
        # Используется итоговый mean: measured_mean нельзя выдавать за полный результат.
        result.add_row(
            Text(LABELS.get(key, key)),
            number(item.get("mean")),
            number(item.get("median")),
            Text(f"{count}/{denominator}", style="yellow" if count < denominator else ""),
        )
    return result


def print_summary(
    report: dict[str, Any],
    output: Path,
    *,
    all_metrics: bool = False,
    no_color: bool = False,
    quiet: bool = False,
    elapsed_seconds: float | None = None,
) -> None:
    if quiet:
        print(output)
        return
    colors = (
        sys.stdout.isatty()
        and not no_color
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )
    console = Console(
        color_system="auto" if colors else None, force_terminal=colors, markup=False, highlight=False
    )
    summary = report["summary"]
    title = "Оценка геометрии"
    if report["purpose"] == "metric_validation":
        title += " · проверка метрик"
    backends = {case["primary"].get("solid_iou", {}).get("backend") for case in report["cases"]} - {None}
    if backends == {"analytic_fixture"}:
        title += " · аналитический IoU"
    console.print(Text(title, style="bold"))
    status = Text(f"Попыток: {summary['attempts']}  ·  ")
    for key, label in (("valid_shape_rate", "VSR"), ("invalidity_ratio", "IR")):
        value = summary[key]
        status.append(f"{label}: {value:.1%}" if value is not None else f"{label}: —")
        status.append("  ·  ")
    status.append(
        "основные метрики рассчитаны" if summary["rankable"] else "основной результат неполный",
        style="green" if summary["rankable"] else "yellow",
    )
    console.print(status)
    config = report["config"]
    tau = config.get("tau_mm")
    threshold = f"{tau:g} мм" if tau is not None else f"{config['tau_relative']:.2%} диагонали эталона"
    console.print(Text(f"F-score: допуск {threshold}. ↑ больше лучше, ↓ меньше лучше.", style="dim"))
    console.print()
    console.print(
        table("Основные метрики", summary["primary"], ["solid_iou", "f_score"], summary["attempts"])
    )
    console.print()
    metrics = summary["secondary"]
    keys = sorted(metrics) if all_metrics else list(BRIEF_SECONDARY)
    if not all_metrics:
        # Не смешиваем реальные кромки и приближение по сетке. Если измерены оба — показываем оба.
        brep_count = metrics.get("edge_chamfer_brep_mm2", {}).get("measured_count", 0)
        proxy_count = metrics.get("edge_chamfer_mesh_proxy_mm2", {}).get("measured_count", 0)
        if not brep_count and proxy_count:
            keys.remove("edge_chamfer_brep_mm2")
        elif not proxy_count:
            keys.remove("edge_chamfer_mesh_proxy_mm2")
    if keys:
        console.print(table("Вспомогательные метрики", metrics, keys, summary["attempts"]))
    if not summary["rankable"]:
        console.print(
            Text("Итог неполный: есть пропуски, ошибки или незавершённые проверки.", style="yellow")
        )
        for case in report["cases"]:
            if case.get("error"):
                console.print(Text(f"{case['id']}: {case['error']}", style="yellow"), soft_wrap=True)
    console.print(Text("Учтено — попытки с оценкой; — означает недоступный итог, а не ноль.", style="dim"))
    if not all_metrics and set(metrics) - set(keys):
        console.print(Text("Все вспомогательные метрики: --all-metrics.", style="dim"))
    for case in report["cases"]:
        reasons = dict.fromkeys(
            value["reason"] for value in case["secondary"].values()
            if value.get("value") is None and value.get("reason")
        )
        for reason in reasons:
            console.print(Text(f"Не рассчитано · {case['id']}: {reason}", style="dim"), soft_wrap=True)
    if elapsed_seconds is not None:
        console.print(Text(f"Общее время прогона: {duration(elapsed_seconds)}", style="bold"))
    console.print(Text(f"Отчёт: {output / 'report.md'}"), soft_wrap=True)
