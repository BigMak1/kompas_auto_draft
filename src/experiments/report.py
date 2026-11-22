"""Независимая офлайн-оценка, агрегация метрик и формирование отчёта HW2."""

import csv
import importlib.metadata
import io
import math
import statistics
from pathlib import Path
from typing import Any

from src.core.config import EvaluationSettings
from src.core.logging import log
from src.core.utils import ROOT, atomic, error_details, read_json, sha256, write_json
from src.evaluation.metrics import evaluate
from src.experiments.plots import geometry_plots, summary_plots
from src.experiments.store import attempts, case_name, read_record, study_path, verify_study

EVALUATOR_FILES = (
    "src/evaluation/metrics.py",
    "src/evaluation/dataset.py",
    "src/core/utils.py",
)
EVALUATOR_PACKAGES = ("numpy", "scipy", "trimesh", "rtree", "manifold3d")


def latest_attempt(directory: Path, job: dict) -> Path | None:
    found = attempts(directory / job["variant"], case_name(job["pair_id"], job["repeat"]))
    return found[-1] if found else None


def verify_evaluator(study: dict, root: Path = ROOT) -> None:
    for rel in EVALUATOR_FILES:
        expected = study["source_sha256"].get(rel)
        current_file = root / rel if (root / rel).is_file() else Path(__file__).resolve().parents[2] / rel
        if sha256(current_file) != expected:
            raise ValueError(f"Код оценщика {rel} отличается от зафиксированного в серии")
    for pkg in EVALUATOR_PACKAGES:
        if importlib.metadata.version(pkg) != study["versions"].get(pkg):
            raise ValueError(f"Версия зависимости {pkg} отличается от зафиксированной в серии")


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
            raise ValueError("Чтение находится вне папки серии")
        read_record(parent)
        if sha256(parent / "record.json") != record["reading_sha256"]:
            raise ValueError("Промежуточный этап чтения был изменён")
    return record


def classify_outcome(record: dict | None, score: dict | None, reading_record: dict | None = None) -> str:
    if record is None or record.get("status") == "started":
        return "infrastructure"
    if record["status"] == "success":
        if score is None or score.get("status") == "error":
            return "infrastructure"
        return "valid_stl" if score.get("status") == "ok" else "invalid_stl"
    if record.get("failure_kind") != "generation":
        return "infrastructure"
    if record.get("stage") == "cad":
        return "cad_error"
    reason = record.get("failure_reason") or (reading_record or {}).get("failure_reason")
    if reason == "max_output_tokens":
        return "token_limit"
    return "schema_or_json"


def evaluate_study(root: Path, name: str) -> dict[str, dict]:
    """Офлайн-оценка всех ячеек серии без OpenAI, CadQuery и MLflow."""
    directory = study_path(root, name)
    study = verify_study(directory)
    verify_evaluator(study, root)
    log("ПРОВЕРКА", "Снимок серии, код оценщика и зависимости совпадают. Запросов к API нет.")
    settings = EvaluationSettings(**study["config"]["evaluation"])
    items = {item["pair_id"]: item for item in study["manifest"]["items"]}
    variants_cfg = study["config"]["variants"]

    all_scores: dict[str, dict] = {}
    for index, job in enumerate(study["schedule"], 1):
        trial = latest_attempt(directory, job)
        if trial is None:
            continue
        identity = trial.relative_to(directory).as_posix()
        destination = trial / "evaluation.json"
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
        except Exception as error:  # noqa: BLE001 — ошибка целостности не даёт ложных нулей.
            result = {
                "status": "error",
                "iou": None,
                "f_score": None,
                "error": error_details(error, "целостность"),
            }
        result["study_fingerprint"] = study["fingerprint"]
        result["record_sha256"] = sha256(trial / "record.json") if (trial / "record.json").is_file() else None
        write_json(destination, result)
        atomic(destination.with_suffix(".sha256"), (sha256(destination) + "\n").encode("utf-8"))
        all_scores[identity] = result
        label = variants_cfg[job["variant"]]["label"]
        if result.get("iou") is not None and result.get("f_score") is not None:
            log(
                "МЕТРИКИ",
                f"{label} · {job['pair_id']}: IoU={result['iou']:.6f}, F-score={result['f_score']:.6f}.",
            )
        else:
            reason = (result.get("error") or {}).get("message", result.get("reason", "Нет измерения"))
            log("ВНИМАНИЕ", f"{job['pair_id']}: оценка отсутствует — {reason}")

    out_dir = root / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "evaluations.json", all_scores)
    return all_scores


def observations(root: Path, directory: Path, study: dict) -> list[dict]:
    verify_evaluator(study, root)
    evaluator_file = (
        root / "src/evaluation/metrics.py"
        if (root / "src/evaluation/metrics.py").is_file()
        else Path(__file__).resolve().parents[1] / "evaluation/metrics.py"
    )
    evaluator_hash = sha256(evaluator_file)
    variants_cfg = study["config"]["variants"]

    rows: list[dict[str, Any]] = []
    for job in sorted(study["schedule"], key=lambda j: (j["variant"], j["pair_id"], j["repeat"])):
        row: dict[str, Any] = {
            **job,
            "attempt": None,
            "status": "missing",
            "outcome_category": "infrastructure",
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
        trial = latest_attempt(directory, job)
        if trial is None:
            rows.append(row)
            continue
        row["attempt"] = trial.relative_to(directory).as_posix()
        try:
            record = verify_trial(directory, trial, study, job, root)
            reading_rec = (
                read_record(directory / record["reading_path"]) if record.get("reading_path") else None
            )
            row.update(
                status=record["status"],
                seconds=record.get("seconds"),
                reading_path=record.get("reading_path"),
            )
            row.update({k: (record.get("usage") or {}).get(k) for k in ("input_tokens", "output_tokens")})
            error = record.get("error") or {}
            row["reason"] = error.get("summary", error.get("message", "Оценка отсутствует"))
            row["outcome_category"] = classify_outcome(record, None, reading_rec)

            score_path = trial / "evaluation.json"
            if not score_path.is_file():
                rows.append(row)
                continue
            score_checksum = score_path.with_suffix(".sha256")
            if (
                not score_checksum.is_file()
                or sha256(score_path) != score_checksum.read_text(encoding="utf-8").strip()
            ):
                raise ValueError("Сохранённая оценка изменена: выполните evaluate")
            score = read_json(score_path)
            if (
                score.get("record_sha256") != sha256(trial / "record.json")
                or score.get("study_fingerprint") != study["fingerprint"]
            ):
                raise ValueError("Устаревшая оценка: другой журнал или конфигурация")
            if score["status"] not in {"ok", "generation_failed", "invalid_prediction"}:
                row["reason"] = (score.get("error") or {}).get(
                    "message", score.get("reason", "Ошибка измерителя")
                )
                row["outcome_category"] = classify_outcome(record, score, reading_rec)
                rows.append(row)
                continue
            if score.get("evaluator_sha256") != evaluator_hash:
                raise ValueError("Код оценщика изменён: выполните evaluate")
            if any(score.get("parameters", {}).get(k) != v for k, v in study["config"]["evaluation"].items()):
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
                outcome_category=classify_outcome(record, score, reading_rec),
                reason="" if score["status"] == "ok" else row["reason"],
            )
        except Exception as error:  # noqa: BLE001 — не включаем повреждённый артефакт в среднее.
            row.update(
                status="integrity_error",
                outcome_category="infrastructure",
                reason=str(error),
                iou=None,
                f_score=None,
                valid=None,
            )
        rows.append(row)

    # Абляции с общим этапом passport должны читать один и тот же паспорт.
    for variant_key, v_cfg in variants_cfg.items():
        base_key = v_cfg.get("ablation_of")
        if not base_key or base_key not in variants_cfg:
            continue
        base_cfg = variants_cfg[base_key]
        if v_cfg["prompts"].get("passport") != base_cfg["prompts"].get("passport") or not v_cfg[
            "prompts"
        ].get("passport"):
            continue
        for pair in study["pairs"]:
            for repeat in range(1, study["repeats"] + 1):
                paired = [
                    r
                    for r in rows
                    if r["pair_id"] == pair
                    and r["repeat"] == repeat
                    and r["variant"] in {variant_key, base_key}
                ]
                if (
                    len(paired) == 2
                    and all(r["reading_path"] for r in paired)
                    and paired[0]["reading_path"] != paired[1]["reading_path"]
                ):
                    for r in paired:
                        r.update(
                            iou=None,
                            f_score=None,
                            valid=None,
                            outcome_category="infrastructure",
                            reason="Абляция и база использовали разные паспорта",
                        )
    return rows


def aggregate(rows: list[dict], study: dict) -> dict:
    variants_cfg = study["config"]["variants"]
    summary: dict[str, Any] = {
        "name": study["name"],
        "split": study["split"],
        "formal_validation": study["formal_validation"],
        "expected": len(study["schedule"]),
        "measured": sum(r["iou"] is not None for r in rows),
        "variants": {},
        "comparisons": [],
    }
    summary["complete"] = summary["measured"] == summary["expected"]

    for variant in variants_cfg:
        subset = [r for r in rows if r["variant"] == variant]
        measured = [r for r in subset if r["iou"] is not None]
        full = len(measured) == len(study["pairs"]) * study["repeats"]
        outcomes: dict[str, int] = {}
        for r in subset:
            cat = r["outcome_category"]
            outcomes[cat] = outcomes.get(cat, 0) + 1
        entry: dict[str, Any] = {
            "measured": len(measured),
            "expected": len(subset),
            "iou": None,
            "iou_part_std": None,
            "f_score": None,
            "f_score_part_std": None,
            "validity": None,
            "per_part": {},
            "repeat_means": {},
            "repeat_std": {},
            "generation_failures": sum(
                r["status"] == "failed" and r["outcome_category"] != "infrastructure" for r in subset
            ),
            "infra_failures": sum(r["outcome_category"] == "infrastructure" for r in subset),
            "outcome_counts": outcomes,
        }
        if full and measured:
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
                part_means = [v[key]["mean"] for v in entry["per_part"].values()]
                entry[key] = statistics.mean(part_means)
                entry[f"{key}_part_std"] = statistics.stdev(part_means) if len(part_means) > 1 else None
                rep_means = [
                    statistics.mean(r[key] for r in measured if r["repeat"] == rep)
                    for rep in range(1, study["repeats"] + 1)
                ]
                entry["repeat_means"][key] = rep_means
                entry["repeat_std"][key] = statistics.stdev(rep_means) if len(rep_means) > 1 else None
            entry["validity"] = statistics.mean(float(r["valid"]) for r in measured)
        for key in ("seconds", "input_tokens", "output_tokens"):
            values = [r[key] for r in subset]
            entry[f"{key}_mean"] = (
                statistics.mean(values) if values and all(v is not None for v in values) else None
            )
        summary["variants"][variant] = entry

    for comp in study["config"].get("comparisons", []):
        cand, ctrl = comp["candidate"], comp["control"]
        a, b = summary["variants"].get(cand), summary["variants"].get(ctrl)
        if not a or not b or a["iou"] is None or b["iou"] is None:
            summary["comparisons"].append(
                {
                    "candidate": cand,
                    "control": ctrl,
                    "question": comp.get("question", ""),
                    "delta_iou": None,
                    "delta_f_score": None,
                    "delta_validity": None,
                    "conclusion": "Недостаточно данных (неполная оценка)",
                }
            )
            continue
        di = a["iou"] - b["iou"]
        df = a["f_score"] - b["f_score"]
        dv = (
            (a["validity"] - b["validity"])
            if a["validity"] is not None and b["validity"] is not None
            else None
        )
        if df > 0 and di >= 0 and (dv is None or dv >= 0):
            conclusion = "Критерий улучшения выполнен (IoU, F-score и валидность не хуже базы)"
        elif di * df < 0 or (df > 0 and dv is not None and dv < 0):
            conclusion = "Компромисс между метриками или валидностью"
        else:
            conclusion = "Без прироста по средним метрикам"
        summary["comparisons"].append(
            {
                "candidate": cand,
                "control": ctrl,
                "question": comp.get("question", ""),
                "delta_iou": di,
                "delta_f_score": df,
                "delta_validity": dv,
                "conclusion": conclusion,
            }
        )
    return summary


def physical_usage(directory: Path) -> dict:
    calls: list[dict | None] = []
    elapsed: list[float | None] = []
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
        "seconds": sum(elapsed) if all(v is not None for v in elapsed) else None,
        **{
            key: sum(u[key] for u in calls) if all(u is not None for u in calls) else None
            for key in ("input_tokens", "output_tokens")
        },
        "note": "Общий этап чтения учтён физически один раз; включены все технические попытки.",
    }


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def report_study(root: Path, name: str) -> dict:
    """Формирует runs.csv, summary.json, report.md и графики в results/<study>/."""
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
    if rows:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic(output / "runs.csv", buffer.getvalue().encode("utf-8"))
    write_json(output / "summary.json", summary)

    cfg = study["config"]
    variants_cfg = cfg["variants"]
    role_names = {"baseline": "Baseline", "improvement": "Улучшение", "ablation": "Абляция"}
    val_label = "формальная валидация val" if study["formal_validation"] else "отладочный/частичный прогон"
    completeness = (
        "**полная оценка**" if summary["complete"] else "**неполная оценка (выводы предварительные)**"
    )
    lines = [
        f"# {cfg.get('title', name)}",
        "",
        "## 1. Условия исследования",
        "",
        f"- **Серия:** `{name}` (`{cfg['name']}`)",
        f"- **Выборка:** `{study['split']}` ({len(study['pairs'])} деталей × {study['repeats']} "
        f"повторов = {summary['expected']} прогонов)",
        f"- **Статус полноты:** {summary['measured']}/{summary['expected']} измерено — {completeness}",
        f"- **Тип запуска:** {val_label}",
        f"- **Модель и бюджет:** `{cfg['agent']['model']}` "
        f"(`reasoning={cfg['agent']['reasoning_effort']}`), "
        f"{cfg['agent']['timeout_seconds']} с и {cfg['agent']['max_output_tokens']} "
        f"выходных токенов на маршрут (чтение ≤ {cfg['reading']['max_output_tokens']})",
        f"- **Воспроизводимость:** `order_seed={cfg['execution']['order_seed']}`, "
        f"`eval_seed={cfg['evaluation']['seed']}`, `dataset_version={cfg['dataset_version']}`",
        "",
    ]
    if cfg.get("hypothesis"):
        lines += [f"> **Общая гипотеза:** {cfg['hypothesis'].strip()}", ""]

    lines += [
        "## 2. Таблица экспериментов",
        "",
        "| Вариант | Роль | Гипотеза | Измерено | IoU | F-score | Валидность | Время, с | Токены (вых) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_key, res in summary["variants"].items():
        v_cfg = variants_cfg[variant_key]
        hyp = " ".join((v_cfg.get("hypothesis") or "").split())
        role_title = role_names.get(v_cfg["role"], v_cfg["role"])
        lines.append(
            f"| {v_cfg['label']} (`{variant_key}`) | {role_title} | {hyp} | "
            f"{res['measured']}/{res['expected']} | {_fmt(res['iou'])} | {_fmt(res['f_score'])} | "
            f"{_fmt(res['validity'])} | {_fmt(res['seconds_mean'])} | {_fmt(res['output_tokens_mean'])} |"
        )

    lines += [
        "",
        "## 3. Абляции и парные сравнения",
        "",
        "| Сравнение | Проверяемый вопрос | Δ IoU | Δ F-score | Δ Валидность | Вывод |",
        "|---|---|---:|---:|---:|---|",
    ]
    for comp in summary["comparisons"]:
        a_lbl = variants_cfg[comp["candidate"]]["label"]
        b_lbl = variants_cfg[comp["control"]]["label"]
        q = " ".join((comp.get("question") or "").split())
        lines.append(
            f"| {a_lbl} − {b_lbl} | {q} | {_fmt(comp['delta_iou'], signed=True)} | "
            f"{_fmt(comp['delta_f_score'], signed=True)} | {_fmt(comp['delta_validity'], signed=True)} | "
            f"{comp['conclusion']} |"
        )

    passport_variants = [
        f"`{key}`" for key, v_cfg in variants_cfg.items() if v_cfg["prompts"].get("passport")
    ]
    shared_passport_line = (
        f"- **Общий паспорт для абляции:** {', '.join(passport_variants[:-1])} и "
        f"{passport_variants[-1]} используют один и тот же "
        "артефакт `passport.json` внутри пары `(pair_id, repeat)`."
        if len(passport_variants) >= 2
        else "- **Общий этап чтения:** повторное использование артефактов чтения "
        "дедуплицируется внутри пары `(pair_id, repeat)`."
    )
    lines += [
        "",
        "## 4. Честное сравнение и контроль утечек",
        "",
        "- **Единая разбивка и отсутствие утечек:** семьи деталей (`family`) и SHA256 файлов `dev`/`val` "
        "не пересекаются; при генерации доступны только PNG листа и ручная подсказка ориентации.",
        f"- **Единый бюджет и модель:** все {len(variants_cfg)} маршрута запускаются на одной модели, "
        f"одном лимите времени ({cfg['agent']['timeout_seconds']:g} с) и выходных токенов "
        f"({cfg['agent']['max_output_tokens']}; в двухэтапных маршрутах бюджет чтения вычитается из "
        "бюджета построителя).",
        shared_passport_line,
        "- **Единый знаменатель:** ошибка генерации (JSON, схема, лимит токенов, невалидный CAD) даёт "
        "`0.0`; инфраструктурный сбой считается пропуском и блокирует признание серии полной.",
        "",
        "## 5. Результаты по деталям",
        "",
        "| Вариант | Деталь | IoU (mean ± std) | F-score (mean ± std) |",
        "|---|---|---:|---:|",
    ]
    for variant_key, res in summary["variants"].items():
        v_lbl = variants_cfg[variant_key]["label"]
        for pair, metrics in res["per_part"].items():
            iou_s = f"{_fmt(metrics['iou']['mean'])} ± {_fmt(metrics['iou']['std'])}"
            f_s = f"{_fmt(metrics['f_score']['mean'])} ± {_fmt(metrics['f_score']['std'])}"
            lines.append(f"| {v_lbl} | {pair} | {iou_s} | {f_s} |")

    actual = summary["physical_usage"]
    lines += [
        "",
        "## 6. Ресурсы и визуализации",
        "",
        f"Фактически выполнено этапов: {actual['attempted_stages']} "
        f"(расход известен для {actual['stages_with_usage']}). "
        f"Суммарный физический расход: вход {_fmt(actual['input_tokens'])} токенов, "
        f"выход {_fmt(actual['output_tokens'])} токенов, время {_fmt(actual['seconds'])} с.",
        "",
        "![Метрики](metrics.png)",
        "",
        "![Метрики по деталям](per_part_metrics.png)",
        "",
        "![Затраты](resources.png)",
        "",
        "![Структура исходов и ошибок](errors_breakdown.png)",
        "",
        "## 7. Ошибки и ограничения",
        "",
        "Полная таблица по каждой ячейке матрицы: [runs.csv](runs.csv).",
    ]
    for row in rows:
        if row["reason"]:
            item_id = f"{row['variant']}/{row['pair_id']}/r{row['repeat']}"
            lines.append(f"- `{item_id}` ({row['outcome_category']}): {row['reason']}")

    summary_plots(rows, summary, study, output)
    geom_files = geometry_plots(root, directory, study, rows, output)
    if geom_files:
        lines += [
            "",
            "## 8. Геометрия: сечения первого повтора",
            "",
            "Серый контур — эталон, синий — сгенерированный STL (сечения на 25%, 50%, 75% габарита).",
        ]
        for filename in geom_files:
            lines += ["", f"![{filename.removeprefix('geometry-').removesuffix('.png')}]({filename})"]

    atomic(output / "report.md", ("\n".join(lines) + "\n").encode("utf-8"))
    log("ОТЧЁТ", f"Сохранён в {output}")
    return summary
