"""Независимая офлайн-оценка, полная матрица результатов и отчёт исследования."""

import csv
import importlib.metadata
import io
import math
import statistics
from pathlib import Path

from src.config import EvaluationSettings
from src.evaluate import evaluate
from src.experiments.config import LABELS
from src.experiments.logging import log
from src.experiments.store import attempts, case_name, read_record, study_path, verify_study
from src.utils import atomic, error_details, read_json, sha256, write_json


def latest(directory: Path, job: dict) -> Path | None:
    paths = attempts(directory / job["variant"], case_name(job["pair_id"], job["repeat"]))
    return paths[-1] if paths else None


def verify_evaluator(study: dict) -> None:
    for name in ("evaluate.py", "config.py", "utils.py"):
        expected = study["source_sha256"].get("src/" + name)
        if sha256(Path(__file__).resolve().parents[1] / name) != expected:
            raise ValueError("Код оценщика отличается от серии; восстановите его из source.zip")
    for name in ("numpy", "scipy", "trimesh", "rtree", "manifold3d"):
        if importlib.metadata.version(name) != study["versions"].get(name):
            raise ValueError("Зависимости оценщика отличаются от серии; используйте сохранённый uv.lock")


def verify_trial(directory: Path, trial: Path, study: dict, job: dict, root: Path) -> dict:
    record = read_record(trial)
    if any(record.get(key) != job[key] for key in ("variant", "pair_id", "repeat")):
        raise ValueError("Журнал относится к другой ячейке матрицы")
    item = next(item for item in study["manifest"]["items"] if item["pair_id"] == job["pair_id"])
    data = (root / study["config"]["manifest"]).parent
    for field in ("input_png", "target_mesh"):
        if sha256(data / item[field]["path"]) != item[field]["sha256"]:
            raise ValueError("Исходные данные изменены после фиксации серии")
    if record["input"]["sha256"] != item["input_png"]["sha256"]:
        raise ValueError("Модель получила другой входной PNG")
    if record.get("reading_path"):
        parent = (directory / record["reading_path"]).resolve()
        if not parent.is_relative_to(directory.resolve()):
            raise ValueError("Чтение находится вне серии")
        read_record(parent)
        if sha256(parent / "record.json") != record["reading_sha256"]:
            raise ValueError("Промежуточный этап изменён")
    return record


def evaluate_study(root: Path, name: str) -> None:
    directory = study_path(root, name)
    study = verify_study(directory)
    verify_evaluator(study)
    log("ПРОВЕРКА", "Снимок серии, версия оценщика и зависимости совпадают. Запросов к API нет.")
    settings = EvaluationSettings(**study["config"]["evaluation"])
    items = {item["pair_id"]: item for item in study["manifest"]["items"]}
    for index, job in enumerate(study["schedule"], 1):
        trial = latest(directory, job)
        if trial is None:
            continue
        identity = trial.relative_to(directory)
        destination = root / "results" / name / identity / "evaluation.json"
        log("ОЦЕНКА", f"{index}/{len(study['schedule'])} · {identity}")
        try:
            record = verify_trial(directory, trial, study, job, root)
            target = items[job["pair_id"]]["target_mesh"]
            result = evaluate(
                trial / "model.stl",
                (root / study["config"]["manifest"]).parent / target["path"],
                settings,
                record,
                expected_reference_sha256=target["sha256"],
            )
        except Exception as error:  # noqa: BLE001 — подмена и отсутствие артефактов не дают нули.
            result = {
                "status": "error",
                "iou": None,
                "f_score": None,
                "error": error_details(error, "целостность"),
            }
        result["study_fingerprint"] = study["fingerprint"]
        result["record_sha256"] = sha256(trial / "record.json") if (trial / "record.json").is_file() else None
        write_json(destination, result)
        atomic(destination.with_suffix(".sha256"), (sha256(destination) + "\n").encode())
        if result.get("iou") is not None and result.get("f_score") is not None:
            log(
                "МЕТРИКИ",
                f"{LABELS[job['variant']]} · {job['pair_id']}: "
                f"IoU={result['iou']:.6f}, F-score={result['f_score']:.6f}.",
            )
        else:
            reason = (result.get("error") or {}).get("message", result.get("reason", "Нет измерения"))
            log("ВНИМАНИЕ", f"{job['pair_id']}: оценка отсутствует — {reason}")


def observations(root: Path, directory: Path, study: dict) -> list[dict]:
    verify_evaluator(study)
    rows = []
    for job in sorted(study["schedule"], key=lambda j: (j["variant"], j["pair_id"], j["repeat"])):
        row = {
            **job,
            "attempt": None,
            "status": "missing",
            "iou": None,
            "f_score": None,
            "valid": None,
            "seconds": None,
            "input_tokens": None,
            "output_tokens": None,
            "tau_mm": None,
            "reason": "Запуск отсутствует",
            "reading_path": None,
        }
        trial = latest(directory, job)
        if trial is None:
            rows.append(row)
            continue
        row["attempt"] = trial.relative_to(directory).as_posix()
        try:
            record = verify_trial(directory, trial, study, job, root)
            row.update(
                status=record["status"],
                seconds=record.get("seconds"),
                reading_path=record.get("reading_path"),
            )
            row.update(
                {key: (record.get("usage") or {}).get(key) for key in ("input_tokens", "output_tokens")}
            )
            error = record.get("error") or {}
            row["reason"] = error.get("summary", error.get("message", "Оценка отсутствует"))
            score_path = root / "results" / study["name"] / row["attempt"] / "evaluation.json"
            if not score_path.is_file():
                rows.append(row)
                continue
            score = read_json(score_path)
            if sha256(score_path) != score_path.with_suffix(".sha256").read_text().strip():
                raise ValueError("Сохранённая оценка изменена: выполните evaluate")
            if (
                score.get("record_sha256") != sha256(trial / "record.json")
                or score.get("study_fingerprint") != study["fingerprint"]
            ):
                raise ValueError("Устаревшая оценка: другой журнал или конфигурация")
            if score["status"] not in {"ok", "generation_failed", "invalid_prediction"}:
                row["reason"] = (score.get("error") or {}).get(
                    "message", score.get("reason", "Ошибка измерителя")
                )
                rows.append(row)
                continue
            if score.get("evaluator_sha256") != sha256(Path(__file__).resolve().parents[1] / "evaluate.py"):
                raise ValueError("Код оценщика изменён: выполните evaluate")
            if any(
                score.get("parameters", {}).get(key) != value
                for key, value in study["config"]["evaluation"].items()
            ):
                raise ValueError("Параметры измерения изменены")
            for metric in ("iou", "f_score"):
                value = score[metric]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Некорректная метрика")
            row.update(
                iou=score["iou"],
                f_score=score["f_score"],
                tau_mm=score.get("tau_mm"),
                valid=score["status"] == "ok",
                reason="" if score["status"] == "ok" else row["reason"],
            )
        except Exception as error:  # noqa: BLE001 — не включаем изменённый артефакт в среднее.
            row.update(status="integrity_error", reason=str(error), iou=None, f_score=None, valid=None)
        rows.append(row)
    # Абляция PNG должна сравнивать один и тот же паспорт, включая технические перезапуски.
    for pair in study["pairs"]:
        for repeat in range(1, study["repeats"] + 1):
            paired = [
                r
                for r in rows
                if r["pair_id"] == pair
                and r["repeat"] == repeat
                and r["variant"] in {"passport_only", "passport_png"}
            ]
            if (
                len(paired) == 2
                and all(r["reading_path"] for r in paired)
                and paired[0]["reading_path"] != paired[1]["reading_path"]
            ):
                for row in paired:
                    row.update(
                        iou=None, f_score=None, valid=None, reason="Маршруты использовали разные паспорта"
                    )
    return rows


def aggregate(rows: list[dict], study: dict) -> dict:
    summary = {
        "name": study["name"],
        "split": study["split"],
        "formal_validation": study["formal_validation"],
        "expected": len(study["schedule"]),
        "measured": sum(r["iou"] is not None for r in rows),
        "variants": {},
    }
    summary["complete"] = summary["measured"] == summary["expected"]
    for variant in study["config"]["variants"]:
        subset = [r for r in rows if r["variant"] == variant]
        measured = [r for r in subset if r["iou"] is not None]
        full = len(measured) == len(study["pairs"]) * study["repeats"]
        entry = {
            "measured": len(measured),
            "expected": len(subset),
            "iou": None,
            "f_score": None,
            "validity": None,
            "per_part": {},
            "repeat_means": {},
            "repeat_std": {},
        }
        if full:
            for pair in study["pairs"]:
                values = [r for r in measured if r["pair_id"] == pair]
                entry["per_part"][pair] = {
                    key: {
                        "mean": statistics.mean(r[key] for r in values),
                        "std": statistics.stdev(r[key] for r in values) if len(values) > 1 else None,
                    }
                    for key in ("iou", "f_score")
                }
            for key in ("iou", "f_score"):
                entry[key] = statistics.mean(v[key]["mean"] for v in entry["per_part"].values())
                means = [
                    statistics.mean(r[key] for r in measured if r["repeat"] == repeat)
                    for repeat in range(1, study["repeats"] + 1)
                ]
                entry["repeat_means"][key] = means
                entry["repeat_std"][key] = statistics.stdev(means) if len(means) > 1 else None
            entry["validity"] = statistics.mean(float(r["valid"]) for r in measured)
        for key in ("seconds", "input_tokens", "output_tokens"):
            values = [r[key] for r in subset]
            entry[key + "_mean"] = statistics.mean(values) if all(v is not None for v in values) else None
        summary["variants"][variant] = entry
    return summary


def physical_usage(directory: Path) -> dict:
    calls = []
    elapsed = []
    for path in directory.glob("*/*/record.json"):
        record = read_record(path.parent)
        if record["kind"] == "reading":
            calls.append(record.get("usage"))
            elapsed.append(record.get("seconds") if record["status"] != "started" else None)
        elif record.get("llm"):
            calls.append(record["llm"].get("usage"))
        elif record.get("llm_started"):
            calls.append(None)
        if record["kind"] == "route":
            shared = 0.0
            if record.get("reading_path"):
                shared = read_record(directory / record["reading_path"]).get("seconds", 0.0)
            elapsed.append(max(0.0, record["seconds"] - shared) if record["status"] != "started" else None)
    return {
        "attempted_stages": len(calls),
        "stages_with_usage": sum(u is not None for u in calls),
        "seconds": sum(elapsed) if all(value is not None for value in elapsed) else None,
        **{
            key: sum(u[key] for u in calls) if all(u is not None for u in calls) else None
            for key in ("input_tokens", "output_tokens")
        },
        "note": "Общий паспорт учтён один раз; включены все сохранённые технические попытки.",
    }


def plots(rows: list[dict], summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(summary["variants"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("iou", "f_score"), strict=True):
        for index, variant in enumerate(variants):
            samples = [r[metric] for r in rows if r["variant"] == variant and r[metric] is not None]
            if samples:
                ax.scatter([index] * len(samples), samples, alpha=0.4, s=20)
            result = summary["variants"][variant]
            if result[metric] is not None:
                ax.scatter(index, result[metric], marker="D", color="black", s=65, zorder=3)
        ax.set(
            xticks=range(len(variants)),
            xticklabels=[LABELS[v].replace(" ", "\n", 1) for v in variants],
            ylim=(-0.03, 1.03),
            ylabel="IoU" if metric == "iou" else "F-score",
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Точки — отдельные результаты; ромб — среднее полной оценки")
    fig.savefig(output / "metrics.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, label in zip(
        axes, ("seconds_mean", "output_tokens_mean"), ("Время, с", "Выходные токены"), strict=True
    ):
        for index, variant in enumerate(variants):
            value = summary["variants"][variant][key]
            if value is not None:
                ax.bar(index, value)
        ax.set(
            xticks=range(len(variants)),
            xticklabels=[LABELS[v].replace(" ", "\n", 1) for v in variants],
            ylabel=label,
        )
    fig.savefig(output / "resources.png", dpi=180)
    plt.close(fig)


def geometry_plots(root: Path, directory: Path, study: dict, rows: list[dict], output: Path) -> list[str]:
    """Первые повторы всех деталей: одинаковые сечения, без выбора удачного ответа."""
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh
    from matplotlib.collections import LineCollection

    from src.evaluate import load_mesh

    files = []
    for pair in study["pairs"]:
        selected = [r for r in rows if r["pair_id"] == pair and r["repeat"] == 1]
        if not any(r["valid"] for r in selected):
            continue
        item = next(item for item in study["manifest"]["items"] if item["pair_id"] == pair)
        reference = load_mesh((root / study["config"]["manifest"]).parent / item["target_mesh"]["path"])
        predictions = {
            row["variant"]: load_mesh(directory / row["attempt"] / "model.stl") if row["valid"] else None
            for row in selected
        }
        meshes = [reference, *(mesh for mesh in predictions.values() if mesh is not None)]
        bounds = np.array(
            [np.min([m.bounds[0] for m in meshes], axis=0), np.max([m.bounds[1] for m in meshes], axis=0)]
        )
        fig, axes = plt.subplots(
            len(selected), 3, figsize=(10, 3 * len(selected)), squeeze=False, constrained_layout=True
        )
        for row_index, row in enumerate(selected):
            prediction = predictions[row["variant"]]
            for axis, normal_axis in zip(axes[row_index], (2, 1, 0), strict=True):
                projected = [i for i in range(3) if i != normal_axis]
                for mesh, color in ((reference, "#777777"), (prediction, "#1877b8")):
                    if mesh is None:
                        continue
                    for fraction in (0.25, 0.5, 0.75):
                        origin = reference.bounds.mean(axis=0)
                        origin[normal_axis] = (
                            reference.bounds[0, normal_axis] + fraction * reference.extents[normal_axis]
                        )
                        segments = trimesh.intersections.mesh_plane(mesh, np.eye(3)[normal_axis], origin)
                        if len(segments):
                            axis.add_collection(
                                LineCollection(
                                    segments[:, :, projected], colors=color, alpha=0.65, linewidths=0.8
                                )
                            )
                margin = max(reference.extents) * 0.05
                axis.set(
                    xlim=(bounds[0, projected[0]] - margin, bounds[1, projected[0]] + margin),
                    ylim=(bounds[0, projected[1]] - margin, bounds[1, projected[1]] + margin),
                    xlabel="XYZ"[projected[0]] + ", мм",
                    ylabel="XYZ"[projected[1]] + ", мм",
                    title=LABELS[row["variant"]],
                )
                axis.set_aspect("equal")
                if prediction is None:
                    axis.text(
                        0.5,
                        0.5,
                        "Нет измеренного результата",
                        ha="center",
                        transform=axis.transAxes,
                        fontsize=8,
                    )
        fig.suptitle(
            f"{pair}, повтор 1 · серый — эталон, синий — результат\n"
            "Сечения на 25%, 50%, 75% габарита эталона; совмещение отсутствует"
        )
        filename = f"geometry-{pair}.png"
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        files.append(filename)
    return files


def report_study(root: Path, name: str) -> dict:
    directory = study_path(root, name)
    study = verify_study(directory)
    rows = observations(root, directory, study)
    summary = aggregate(rows, study)
    try:
        summary["physical_usage"] = physical_usage(directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary["physical_usage"] = {
            "attempted_stages": "неизвестно",
            "stages_with_usage": "неизвестно",
            "seconds": None,
            "input_tokens": None,
            "output_tokens": None,
            "error": str(error),
        }
    output = root / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic(output / "runs.csv", buffer.getvalue().encode())
    write_json(output / "summary.json", summary)
    lines = [
        f"# {study['config'].get('title', name)}",
        "",
        "## Кратко",
        "",
        f"Серия: `{name}`.",
        "",
        f"Измерено {summary['measured']}/{summary['expected']}. "
        f"Часть: {study['split']}; повторов: {study['repeats']}.",
        "Полная оценка."
        if summary["complete"]
        else "**Оценка неполная: вывод о сравнении маршрутов не сделан.**",
        "Финальная матрица val."
        if study["formal_validation"]
        else "Отладочная серия; не финальный эксперимент.",
        "",
        "| Маршрут | Измерено | IoU | F-score | Валидность |",
        "|---|---:|---:|---:|---:|",
    ]

    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.6f}"

    for variant, result in summary["variants"].items():
        lines.append(
            f"| {LABELS[variant]} | {result['measured']}/{result['expected']} | "
            f"{fmt(result['iou'])} | {fmt(result['f_score'])} | {fmt(result['validity'])} |"
        )
    lines += [
        "",
        "## Гипотезы и абляции",
        "",
        "| Сравнение | Δ IoU | Δ F-score | Вывод по средним |",
        "|---|---:|---:|---|",
    ]
    comparisons = [
        ("passport_only", "direct"),
        ("passport_png", "direct"),
        ("passport_png", "passport_only"),
        ("passport_png", "notes_png"),
    ]
    for improved, control in comparisons:
        a, b = summary["variants"].get(improved), summary["variants"].get(control)
        if not summary["complete"] or not a or not b:
            lines.append(f"| {LABELS[improved]} − {LABELS[control]} | — | — | Недостаточно результатов |")
            continue
        di, df = a["iou"] - b["iou"], a["f_score"] - b["f_score"]
        passed = df > 0 and di >= 0 and a["validity"] >= b["validity"]
        conclusion = "Критерий по средним выполнен" if passed else "Критерий по средним не выполнен"
        if di * df < 0 or (df > 0 and a["validity"] < b["validity"]):
            conclusion = "Компромисс между метриками или валидностью"
        lines.append(f"| {LABELS[improved]} − {LABELS[control]} | {di:+.6f} | {df:+.6f} | {conclusion} |")
    lines += [
        "",
        "## Детали и разброс повторов",
        "",
        "Среднее ± выборочное стандартное отклонение внутри одной детали; — если повтор один.",
        "",
        "| Маршрут | Деталь | IoU | F-score |",
        "|---|---|---:|---:|",
    ]
    for variant, result in summary["variants"].items():
        for pair, metrics in result["per_part"].items():
            cells = [
                fmt(metrics[key]["mean"]) + " ± " + fmt(metrics[key]["std"]) for key in ("iou", "f_score")
            ]
            lines.append(f"| {LABELS[variant]} | {pair} | {' | '.join(cells)} |")
    lines += [
        "",
        "Повторы не считаются независимыми деталями; статистическая значимость не заявляется.",
        "Полные значения и средние по номерам повторов — в [summary.json](summary.json).",
        "",
        "## Затраты",
        "",
        "| Маршрут | Среднее время, с | Входные токены | Выходные токены |",
        "|---|---:|---:|---:|",
    ]
    for variant, result in summary["variants"].items():
        cells = [fmt(result[key + "_mean"]) for key in ("seconds", "input_tokens", "output_tokens")]
        lines.append(f"| {LABELS[variant]} | {' | '.join(cells)} |")
    actual = summary["physical_usage"]
    lines += [
        "",
        "В таблице общий паспорт включён в каждый использовавший его маршрут. "
        f"Фактически сохранено этапов: {actual['attempted_stages']}; "
        f"с известным расходом: {actual['stages_with_usage']}. "
        f"Всего входных токенов: {fmt(actual['input_tokens'])}; "
        f"выходных: {fmt(actual['output_tokens'])}; время этапов и построений: {fmt(actual['seconds'])} с. "
        "Здесь общий паспорт учтён один раз, а технические попытки включены.",
        "",
        "![Метрики](metrics.png)",
        "",
        "![Затраты](resources.png)",
        "",
        "## Ошибки и ограничения",
        "",
        "Полная таблица, включая отсутствующие результаты и причины: [runs.csv](runs.csv).",
        "Провал генерации — ноль; ошибка инфраструктуры или измерителя — пропуск.",
        "Причины геометрических различий требуют просмотра чертежа, паспорта и модели; "
        "одних метрик недостаточно.",
    ]
    for row in rows:
        if row["reason"]:
            lines.append(f"- {row['variant']}/{row['pair_id']}/r{row['repeat']}: {row['reason']}")
    plots(rows, summary, output)
    files = geometry_plots(root, directory, study, rows, output)
    if files:
        lines += [
            "",
            "## Геометрия: первый повтор каждой детали",
            "",
            "Показаны одинаковые сечения всех маршрутов. Это иллюстрация, не дополнительная метрика; "
            "мелкий элемент между сечениями может быть не виден.",
        ]
        for filename in files:
            lines += ["", f"![{filename.removeprefix('geometry-').removesuffix('.png')}]({filename})"]
    atomic(output / "report.md", ("\n".join(lines) + "\n").encode())
    log("ОТЧЁТ", str(output))
    return summary
