"""Оценка набора: ошибки генератора, измерителя и неполные входы разделены."""

import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from eval.geometry import PROTOCOL_VERSION
from eval.geometry.alignment import align
from eval.geometry.diagnostics import compare_facts, mesh_diagnostics, requirement_metrics
from eval.geometry.io import atomic_json, atomic_text, create_evaluation_directory, digest, read_json, sha256
from eval.geometry.mesh import normalize_vertices
from eval.geometry.models import BooleanResult, Case, Config, Evaluation, Facts, ModelInput
from eval.geometry.progress import ProgressCallback, no_progress, prefix_progress
from eval.geometry.readiness import case_issues
from eval.geometry.solid import evaluate_solid, solid_request
from eval.geometry.surface import cloud_distances, distance_metrics, evaluate_surface, sample_mesh_edges

SCALE_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4}
SOLID_SECONDARY = ("symmetric_difference_mm3", "extra_material_mm3", "missing_material_mm3")


def metric(
    value: float | None, reason: str = "Нет входных данных", status: str = "not_computed"
) -> dict[str, Any]:
    return {
        "value": value,
        "status": "ok" if value is not None else status,
        **({"reason": reason} if value is None else {}),
    }


def load_model(model: ModelInput, root: Path, validation: bool) -> tuple[trimesh.Trimesh, Facts | None, dict]:
    path = (root / model.mesh).resolve()
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Пустая/неконечная сетка: {path}")
    mesh.vertices *= SCALE_MM[model.units]
    mesh, normalization = normalize_vertices(mesh)
    if mesh.area <= 0:
        raise ValueError(f"Нулевая площадь сетки: {path}")
    hashes = {
        "mesh": sha256(path),
        "mesh_path": str(path),
        "units": model.units,
        "mesh_normalization": normalization,
    }
    facts = None
    if model.facts:
        facts_path = (root / model.facts).resolve()
        facts = Facts.model_validate(read_json(facts_path))
        hashes["facts"] = sha256(facts_path)
        if facts.mesh_sha256 != hashes["mesh"]:
            raise ValueError("Снимок R2 относится к другой сетке")
        if facts.mesh_units != model.units:
            raise ValueError("Единицы сетки не совпадают со снимком R2")
        if facts.source != "kompas" and not validation:
            raise ValueError("Аналитические факты разрешены только для metric_validation")
        hashes["native"] = facts.native_sha256
    if model.native:
        native_path = (root / model.native).resolve()
        native_hash = sha256(native_path)
        if facts and facts.native_sha256 and facts.native_sha256 != native_hash:
            raise ValueError("Снимок R2 относится к другому CAD-документу")
        hashes.update(native=native_hash, native_path=str(native_path))
    if model.edge_points:
        hashes["edge_points"] = sha256((root / model.edge_points).resolve())
    if model.export_manifest:
        hashes["export_manifest"] = sha256(root / model.export_manifest)
    return mesh, facts, hashes


def copy_guard(
    prediction: Facts | None, reference: Facts | None, pred_hashes: dict, ref_hashes: dict
) -> dict:
    checks = {
        "native_sha256": (pred_hashes.get("native"), ref_hashes.get("native")),
        "tree_fingerprint": (
            prediction.tree_fingerprint if prediction else None,
            reference.tree_fingerprint if reference else None,
        ),
        "document_fingerprint": (
            prediction.document_fingerprint if prediction else None,
            reference.document_fingerprint if reference else None,
        ),
    }
    equal = [name for name, (pred, ref) in checks.items() if pred and ref and pred == ref]
    missing = [name for name, (pred, ref) in checks.items() if not pred or not ref]
    return {
        "status": "rejected" if equal else ("incomplete" if missing else "clear"),
        "matching": equal,
        "missing": missing,
    }


def validity(facts: Facts | None, config: Config) -> bool | None:
    if facts is None:
        return None
    if (
        facts.solid_valid is False
        or facts.volume_mm3 == 0
        or (facts.body_count is not None and facts.body_count != config.expected_bodies)
    ):
        return False
    if facts.solid_valid is True and facts.volume_mm3 is not None and facts.body_count is not None:
        return True
    return None


def scalar_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    output = {}
    for name, value in data.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            output.update(scalar_metrics(value, f"{key}."))
        elif value is None:
            output[key] = metric(None)
        elif isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
            output.update({f"{key}.{index}": metric(float(item)) for index, item in enumerate(value)})
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            output[key] = metric(float(value))
    return output


def solid_secondary(solid: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"solid.{name}": metric(
            solid.get(name), solid.get("reason", "Нет результата пересечения тел"), solid["status"]
        )
        for name in SOLID_SECONDARY
    }


def explain_missing_metrics(
    report: dict[str, Any], case: Case, prediction: Facts | None, reference: Facts | None
) -> None:
    """Причины пропусков дополняют числа, не меняя правила допуска или формулы."""
    reasons = {}
    if not case.requirements:
        requirements_reason = "В evaluation.json не задан список требований чертежа (requirements)"
    else:
        requirements_reason = "Нет требований или именованных измерений нужного типа для этого показателя"
    for key in report["secondary"]:
        if key.startswith("dimension_") or key in (
            "specification_compliance", "g_csr_project",
        ):
            reasons[key] = requirements_reason
    if prediction is None or reference is None:
        mass_reason = inertia_reason = "Нет снимка измерений ядра для обеих моделей"
    else:
        if not prediction.material or not reference.material:
            mass_reason = "Не указан материал в снимках измерений; сопоставимость масс не подтверждена"
        elif prediction.material != reference.material:
            mass_reason = "Материалы моделей различаются; масса и массовые моменты не сравниваются"
        else:
            mass_reason = "Нет корректных значений массы обеих моделей"
        if prediction.inertia_unit == reference.inertia_unit == "g_mm2" and (
            not prediction.material or not reference.material or prediction.material != reference.material
        ):
            inertia_reason = mass_reason
        else:
            inertia_reason = "Нет главных моментов обеих моделей в совместимых единицах"
    reasons["facts.mass_g_relative_error"] = mass_reason
    reasons["facts.principal_moments_relative_error"] = inertia_reason
    for key, reason in reasons.items():
        value = report["secondary"].get(key)
        if value is not None and value["value"] is None:
            value["reason"] = reason


def evaluate_case(
    case: Case,
    root: Path,
    config: Config,
    results: dict[str, BooleanResult],
    validation: bool,
    artifacts: Path,
    progress: ProgressCallback = no_progress,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {
        "id": case.id,
        "family": case.family,
        "tags": case.tags,
        "generation_status": case.generation_status,
        "valid_shape": None,
        "eligible": False,
        "primary": {},
        "secondary": {},
    }
    requests: list[dict[str, Any]] = []
    try:
        progress("Чтение сетки и измерений эталона")
        reference, ref_facts, ref_hashes = load_model(case.reference, root, validation)
        if validity(ref_facts, config) is False:
            raise ValueError("Невалидный эталон; это ошибка набора, не генератора")
        report["inputs"] = {"reference": ref_hashes}
        if case.generation_status != "success":
            report.update(valid_shape=False, eligible=validity(ref_facts, config) is True, status="invalid")
            report["primary"] = {name: metric(0.0) for name in ("solid_iou", "f_score")}
            for value in report["primary"].values():
                value["reason"] = "Ноль за невалидную генерацию"
            report["secondary"]["siou_cadbench_formula_p2p"] = metric(0.0)
            report["requirements"] = requirement_metrics(case.requirements, {})
            report["secondary"].update(
                scalar_metrics({k: v for k, v in report["requirements"].items() if k != "checks"})
            )
            explain_missing_metrics(report, case, None, ref_facts)
            return report, requests
        assert case.prediction is not None
        progress("Чтение сетки и измерений предсказания")
        prediction, pred_facts, pred_hashes = load_model(case.prediction, root, validation)
        report["inputs"]["prediction"] = pred_hashes
        guard = copy_guard(pred_facts, ref_facts, pred_hashes, ref_hashes)
        report["copy_check"] = {**guard, "validation_exemption": validation}
        if guard["status"] == "rejected" and not validation:
            report["status"] = "rejected_copy"
            report["primary"] = {
                name: metric(None, "Копия эталона", "rejected") for name in ("solid_iou", "f_score")
            }
            return report, requests
        report["valid_shape"] = validity(pred_facts, config)
        report["eligible"] = (
            validity(ref_facts, config) is True
            and report["valid_shape"] is not None
            and (validation or guard["status"] == "clear")
        )
        if report["valid_shape"] is False:
            report["status"] = "invalid"
            report["primary"] = {name: metric(0.0) for name in ("solid_iou", "f_score")}
            report["secondary"]["siou_cadbench_formula_p2p"] = metric(0.0)
        if report["valid_shape"] is False:
            matrix, alignment = np.eye(4), {"method": "not_run_invalid_generation"}
        else:
            matrix, alignment = align(prediction, reference, pred_facts, ref_facts, config, progress)
        report["alignment"] = {**alignment, "prediction_to_reference": matrix.tolist()}
        moved = prediction.copy()
        moved.apply_transform(matrix)
        length = float(np.linalg.norm(reference.extents))
        report["status"] = "invalid" if report["valid_shape"] is False else "evaluated"
        if report["valid_shape"] is not False:
            for pose, mesh, transform in (("aligned", moved, matrix), ("unaligned", prediction, np.eye(4))):
                label = "после совмещения" if pose == "aligned" else "исходная поза"
                surface, deviations = evaluate_surface(
                    mesh, reference, config, length, prefix_progress(progress, label)
                )
                request = solid_request(
                    case.id,
                    pose,
                    pred_hashes.get("native"),
                    ref_hashes.get("native"),
                    pred_hashes["mesh"],
                    ref_hashes["mesh"],
                    transform,
                    prediction_units=case.prediction.units,
                    reference_units=case.reference.units,
                )
                requests.append(request)
                progress(f"IoU и объёмные отклонения: {label}")
                solid = evaluate_solid(request, results, pred_facts, ref_facts, validation)
                report[pose] = {
                    "surface": surface,
                    "solid": solid,
                    "request_sha256": request["request_sha256"],
                }
                if pose == "aligned":
                    report["primary"] = {"solid_iou": solid, "f_score": metric(surface["f_score"])}
                    report["secondary"].update(solid_secondary(solid))
                    report["secondary"].update(scalar_metrics(surface))
                    report["secondary"].pop("f_score", None)
                    for name in ("tau_mm", "reference_length_mm"):
                        report["secondary"].pop(name, None)
                    atomic_json(artifacts / case.id / "deviations.json", deviations)
        pred_diag = mesh_diagnostics(moved, prefix_progress(progress, "Сетка предсказания"))
        ref_diag = mesh_diagnostics(reference, prefix_progress(progress, "Сетка эталона"))
        report["mesh_diagnostics"] = {"prediction": pred_diag, "reference": ref_diag}
        if report["valid_shape"] is not False:
            bad_exports = [
                name
                for name, diagnostic in report["mesh_diagnostics"].items()
                if not diagnostic["watertight"]
                or not diagnostic["winding_consistent"]
                or diagnostic["degenerate_faces"]
                or diagnostic["nonmanifold_edges"]
                or (diagnostic["self_intersection_ratio"].get("value") or 0) > 0
            ]
            if bad_exports:
                report["status"] = "evaluator_error"
                report["error"] = f"Некорректная сетка при оценке тела: {', '.join(bad_exports)}"
            elif any(
                diagnostic["self_intersection_ratio"]["status"] != "ok"
                for diagnostic in report["mesh_diagnostics"].values()
            ):
                report["status"] = "evaluator_error"
                report["error"] = "Не выполнена проверка самопересечений сеток"
        report["secondary"].update(
            scalar_metrics(
                {
                    k: pred_diag[k]
                    for k in ("dangling_edge_length_mm", "flux_enclosure_error_mm2", "flux_vector_norm_mm2")
                }
            )
        )
        report["secondary"]["self_intersection_ratio"] = pred_diag["self_intersection_ratio"]
        report["secondary"]["segment_error"] = metric(
            abs(pred_diag["connected_segments"] - ref_diag["connected_segments"])
            / ref_diag["connected_segments"]
        )
        if case.prediction.edge_points and case.reference.edge_points:
            progress("ECD: чтение точек B-rep-кромок")
            pred_edges = np.asarray(read_json(root / case.prediction.edge_points), dtype=float)
            ref_edges = np.asarray(read_json(root / case.reference.edge_points), dtype=float)
            # Проверяем форму до умножения и применения матрицы.
            cloud_distances(pred_edges, ref_edges)
            pred_edges = trimesh.transform_points(pred_edges * SCALE_MM[case.prediction.units], matrix)
            ref_edges = ref_edges * SCALE_MM[case.reference.units]
            name = "edge_chamfer_brep_mm2"
        else:
            progress("ECD: выборка рёбер сетки предсказания")
            pred_edges = sample_mesh_edges(moved, config, config.seed + 200)
            progress("ECD: выборка рёбер сетки эталона")
            ref_edges = sample_mesh_edges(reference, config, config.seed + 201)
            name = "edge_chamfer_mesh_proxy_mm2"
            report["secondary"]["edge_chamfer_brep_mm2"] = metric(
                None, "Нужны выборки B-rep-кромок обеих моделей"
            )
        progress("ECD: расстояния между кромками")
        report["secondary"][name] = metric(
            distance_metrics(*cloud_distances(pred_edges, ref_edges))["chamfer_l2_mm2"]
            if pred_edges is not None and ref_edges is not None
            else None,
            "Не найдены кромки; пустое множество не означает нулевую ошибку",
        )
        report["facts_comparison"] = compare_facts(pred_facts, ref_facts, matrix, config, progress)
        report["secondary"].update(scalar_metrics(report["facts_comparison"], "facts."))
        progress("Требования чертежа: размеры, ограничения и соответствие спецификации")
        report["requirements"] = requirement_metrics(
            case.requirements,
            pred_facts.measurements if pred_facts and report["valid_shape"] is not False else {},
        )
        report["secondary"].update(
            scalar_metrics({key: value for key, value in report["requirements"].items() if key != "checks"})
        )
        explain_missing_metrics(report, case, pred_facts, ref_facts)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        report["status"] = "evaluator_error"
        report["error"] = f"{type(error).__name__}: {error}"
        for name in ("solid_iou", "f_score"):
            report["primary"].setdefault(name, metric(None, report["error"], "error"))
    return report, requests


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(reports)
    known = [r["valid_shape"] for r in reports if r["valid_shape"] is not None]
    all_eligible = all(
        r["eligible"] and r["status"] not in ("evaluator_error", "rejected_copy") for r in reports
    )
    summary: dict[str, Any] = {
        "attempts": count,
        "validity_known": len(known),
        "valid_shape_rate": sum(known) / count if len(known) == count else None,
        "invalidity_ratio": (count - sum(known)) / count if len(known) == count else None,
        "primary": {},
        "secondary": {},
    }
    for role in ("primary", "secondary"):
        names = sorted({key for report in reports for key in report[role]})
        for name in names:
            values = [
                report[role][name]["value"]
                for report in reports
                if name in report[role]
                and report[role][name]["status"] == "ok"
                and report[role][name]["value"] is not None
                and (
                    role == "primary"
                    or report["status"] != "invalid"
                    or name
                    in (
                        "siou_cadbench_formula_p2p",
                        "specification_compliance",
                        "dimension_compliance",
                        "g_csr_project",
                    )
                )
            ]
            complete = len(values) == count and all_eligible
            mean = float(np.mean(values)) if values else None
            median = float(np.median(values)) if values else None
            summary[role][name] = {
                "mean": mean if role != "primary" or complete else None,
                "median": median if role != "primary" or complete else None,
                "measured_mean": mean,
                "measured_median": median,
                "measured_count": len(values),
                "total_count": count,
                "complete": complete,
            }
    summary["rankable"] = (
        all_eligible
        and len(summary["primary"]) == 2
        and all(item["complete"] for item in summary["primary"].values())
    )
    return summary


def display_number(value: float | None) -> str:
    return f"{value:.6g}" if value is not None else "не считалось"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Оценка CAD-моделей",
        "",
        "## Кратко",
        "",
        "Основные показатели — объёмный IoU и F-score. Остальные — вспомогательные.",
        f"Протокол: `{PROTOCOL_VERSION}`. Назначение: `{report['purpose']}`.",
        f"Полнота основных метрик: {'да' if report['summary']['rankable'] else 'нет'}.",
        "",
        "| Деталь | Статус | IoU ↑ | F-score ↑ |",
        "|---|---|---:|---:|",
    ]
    for row in report["cases"]:
        values = [row["primary"].get(name, {}).get("value") for name in ("solid_iou", "f_score")]
        labels = [f"{value:.6g}" if value is not None else "не считалось / ошибка" for value in values]
        lines.append(f"| {row['id']} | {row['status']} | {labels[0]} | {labels[1]} |")
    lines += ["", "Основные средние по всем попыткам (нули за невалидную генерацию):", ""]
    for name, value in report["summary"]["primary"].items():
        lines.append(
            f"- {name}: {display_number(value['mean'])}; "
            f"измерено {value['measured_count']}/{value['total_count']}."
        )
    lines += [
        "",
        "## Вспомогательные метрики",
        "",
        "| Показатель | Среднее | Медиана | Учтено попыток |",
        "|---|---:|---:|---:|",
    ]
    for name, value in report["summary"]["secondary"].items():
        lines.append(
            f"| {name} | {display_number(value['mean'])} | {display_number(value['median'])} "
            f"| {value['measured_count']} |"
        )
    lines += ["", "## Причины пропусков", ""]
    for case in report["cases"]:
        reasons = dict.fromkeys(
            item["reason"] for item in case["secondary"].values()
            if item.get("value") is None and item.get("reason")
        )
        lines.extend(f"- {case['id']}: {reason}." for reason in reasons)
    lines += [
        "",
        f"VSR: {display_number(report['summary']['valid_shape_rate'])}; "
        f"IR: {display_number(report['summary']['invalidity_ratio'])}.",
        "",
        "Подробности, причины пропусков, исходная поза и преобразования — в `metrics.json`.",
        "Запросы к КОМПАС — в `raw/solid_requests.json`; карты расстояний — в `raw/<id>/deviations.json`.",
        "Аналитический стенд проверяет вычислитель и не является оценкой реконструкции CAD.",
        "",
    ]
    return "\n".join(lines)


def run_evaluation(
    root: Path,
    output: Path | None = None,
    *,
    evaluation: Evaluation | None = None,
    progress: ProgressCallback = no_progress,
) -> Path:
    started = time.perf_counter()
    started_at = datetime.now().astimezone()
    progress("Чтение настроек оценки")
    root = root.resolve()
    input_path = root / "evaluation.json"
    evaluation = evaluation or Evaluation.model_validate(read_json(input_path))
    validation = evaluation.purpose == "metric_validation"
    if output is None:
        output = create_evaluation_directory(root / "evaluation", started_at)
    else:
        output.mkdir(parents=True, exist_ok=False)
    results = read_boolean_results(root, evaluation)
    atomic_json(output / "raw" / "evaluation-input.json", evaluation.model_dump())
    reports, requests = [], []
    for index, case in enumerate(evaluation.cases, start=1):
        case_progress = prefix_progress(progress, f"{case.id} ({index}/{len(evaluation.cases)})")
        case_progress("Проверка входных файлов")
        issues = case_issues(case, root, evaluation)
        if issues:
            reason = "; ".join(f"{item['role']}: {item['message']}" for item in issues)
            reports.append(
                {
                    "id": case.id,
                    "family": case.family,
                    "tags": case.tags,
                    "generation_status": case.generation_status,
                    "valid_shape": None,
                    "eligible": False,
                    "status": "missing_inputs",
                    "error": reason,
                    "input_issues": issues,
                    "primary": {
                        name: metric(None, reason, "not_computed") for name in ("solid_iou", "f_score")
                    },
                    "secondary": {},
                }
            )
            continue
        report, case_requests = evaluate_case(
            case, root, evaluation.config, results, validation, output / "raw", case_progress
        )
        reports.append(report)
        requests.extend(case_requests)
    progress("Формирование таблиц и сохранение отчёта")
    return write_evaluation(root, output, evaluation, reports, requests, started_at, started)


def read_boolean_results(root: Path, evaluation: Evaluation) -> dict[str, BooleanResult]:
    results = {}
    if evaluation.boolean_results:
        for raw in read_json(root / evaluation.boolean_results):
            item = BooleanResult.model_validate(raw)
            if item.request_sha256 in results:
                raise ValueError("Повтор результата для одного запроса КОМПАС")
            results[item.request_sha256] = item
    return results


def write_evaluation(
    root: Path,
    output: Path,
    evaluation: Evaluation,
    reports: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    started_at: datetime,
    started: float,
    reused_from: Path | None = None,
) -> Path:
    input_path = root / "evaluation.json"
    atomic_json(output / "raw/evaluation-input.json", evaluation.model_dump())
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": evaluation.purpose,
        "config": evaluation.config.model_dump(),
        "cases": reports,
        "summary": summarize(reports),
        "by_family": {
            family: summarize([r for r in reports if r["family"] == family])
            for family in sorted({r["family"] for r in reports})
        },
        "by_tag": {
            tag: summarize([r for r in reports if tag in r["tags"]])
            for tag in sorted({tag for r in reports for tag in r["tags"]})
        },
    }
    atomic_json(output / "raw" / "solid_requests.json", requests)
    atomic_json(output / "metrics.json", report)
    atomic_text(output / "report.md", markdown_report(report))
    atomic_text(
        output / "notes.md",
        "# Заметки оценки\n\n## Кратко\n\n"
        "Автоматический результат. Пропуски и ошибки сохранены в metrics.json.\n",
    )
    manifest = {
        "run_id": output.name,
        "date": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "track": "eval",
        "protocol_version": PROTOCOL_VERSION,
        "purpose": evaluation.purpose,
        "input": {"path": str(input_path), "sha256": sha256(input_path)},
        "settings": evaluation.config.model_dump(),
        "export_settings": evaluation.export.model_dump(),
        "resolved_input_sha256": sha256(output / "raw/evaluation-input.json"),
        "metrics_sha256": sha256(output / "metrics.json"),
        "versions": {package: version(package) for package in ("numpy", "scipy", "trimesh", "libigl")},
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "protocol_document_sha256": sha256(Path(__file__).parent / "METRICS.md"),
        "dependencies_lock_sha256": sha256(Path(__file__).parents[2] / "uv.lock"),
        "command": f"uv run poe eval-geometry {root}",
        "cost": {"seconds": time.perf_counter() - started},
        "schema_version": PROTOCOL_VERSION,
        "prompt_version": None,
    }
    if evaluation.boolean_results:
        manifest["boolean_results_sha256"] = sha256(root / evaluation.boolean_results)
        atomic_json(output / "raw" / "boolean-results.json", read_json(root / evaluation.boolean_results))
    if reused_from is not None:
        manifest["reused_geometry"] = {
            "path": str(reused_from),
            "metrics_sha256": sha256(reused_from / "metrics.json"),
            "manifest_sha256": sha256(reused_from / "manifest.json"),
        }
    atomic_json(output / "manifest.json", manifest)
    return output


def finalize_solid_results(
    root: Path,
    preliminary: Path,
    evaluation: Evaluation,
    progress: ProgressCallback = no_progress,
) -> Path:
    """Обновляет только булевы показатели; не вызывает ICP, поверхности и размеры.

    Используется для текущего запуска: конфигурация, входные файлы и код должны
    совпадать с предварительным расчётом. Исторический отчёт не изменяется.
    """
    started = time.perf_counter()
    started_at = datetime.now().astimezone()
    root, preliminary = root.resolve(), preliminary.resolve()
    previous = Evaluation.model_validate(read_json(preliminary / "raw/evaluation-input.json"))
    if previous.model_dump(exclude={"boolean_results"}) != evaluation.model_dump(exclude={"boolean_results"}):
        raise ValueError("Настройки или входы изменились после предварительного расчёта")
    manifest = read_json(preliminary / "manifest.json")
    if (
        sha256(preliminary / "metrics.json") != manifest["metrics_sha256"]
        or sha256(preliminary / "raw/evaluation-input.json") != manifest["resolved_input_sha256"]
    ):
        raise ValueError("Предварительный отчёт или его входное задание изменились")
    sources = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    if manifest["protocol_version"] != PROTOCOL_VERSION or manifest["source_sha256"] != sources:
        raise ValueError("Код оценки изменился после предварительного расчёта")
    if manifest["dependencies_lock_sha256"] != sha256(Path(__file__).parents[2] / "uv.lock"):
        raise ValueError("Зависимости изменились после предварительного расчёта")
    reports = read_json(preliminary / "metrics.json")["cases"]
    cases = {case.id: case for case in evaluation.cases}
    for report in reports:
        for role, hashes in report.get("inputs", {}).items():
            model = getattr(cases[report["id"]], role)
            for field in ("mesh", "facts", "native", "edge_points", "export_manifest"):
                path = getattr(model, field)
                if path and field in hashes and sha256(root / path) != hashes[field]:
                    raise ValueError(f"{report['id']}: файл {role}.{field} изменился во время расчёта")
    requests = read_json(preliminary / "raw/solid_requests.json")
    by_hash = {}
    for request in requests:
        key = request["request_sha256"]
        if digest({k: v for k, v in request.items() if k != "request_sha256"}) != key:
            raise ValueError("Повреждён запрос пересечения предварительного расчёта")
        by_hash[key] = request
    results = read_boolean_results(root, evaluation)
    progress("Обновление IoU и объёмных отклонений; остальные метрики уже рассчитаны")
    for report in reports:
        case = cases[report["id"]]
        facts = {
            role: Facts.model_validate(read_json(root / model.facts)) if model and model.facts else None
            for role in ("prediction", "reference")
            if (model := getattr(case, role)) is not None and role in report.get("inputs", {})
        }
        for pose in ("aligned", "unaligned"):
            if pose not in report:
                continue
            request = by_hash[report[pose]["request_sha256"]]
            solid = evaluate_solid(
                request, results, facts.get("prediction"), facts.get("reference"),
                evaluation.purpose == "metric_validation",
            )
            report[pose]["solid"] = solid
            if pose == "aligned":
                report["primary"]["solid_iou"] = solid
                report["secondary"].update(solid_secondary(solid))
    output = create_evaluation_directory(root / "evaluation", started_at)
    for source in (preliminary / "raw").glob("*/deviations.json"):
        atomic_json(output / "raw" / source.relative_to(preliminary / "raw"), read_json(source))
    progress("Формирование итоговых таблиц без повторного расчёта геометрии")
    return write_evaluation(
        root, output, evaluation, reports, requests, started_at, started, reused_from=preliminary
    )
