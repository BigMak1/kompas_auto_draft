"""Независимая оценка STL: IoU и F-score, без SDK агента и CadQuery.

Формулы из kompas-drawings, eval/geometry/{surface,solid}.py;
ревизия 63f086abfcf22bf3a00cfe1c7bb0aaad220d5208 с локальными изменениями.
SHA256 surface: 9ab3c60318e6c70ab270687868b73269e8b4d1e25b578ac5b0d2e35eb7e8d263;
SHA256 solid: 4fb615c02915c64682d10f14acd37894c7e85cfdfc88ca08d9e6777a64213dc7.
Пересечение треугольных тел выполняет Manifold.
"""

import argparse
import importlib.metadata
import math
from pathlib import Path

import numpy as np
import trimesh

from src.config import ROOT, EvaluationSettings, load_config
from src.utils import Run, atomic, error_details, log, log_error, read_json, select_runs, sha256, write_json


def f_score(forward: np.ndarray, backward: np.ndarray, tau: float) -> float:
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau должен быть положительным конечным числом")
    for distances in (forward, backward):
        if (
            distances.ndim != 1
            or not len(distances)
            or not np.isfinite(distances).all()
            or np.any(distances < 0)
        ):
            raise ValueError("Расстояния должны быть непустыми, конечными и неотрицательными")
    precision, recall = float(np.mean(forward <= tau)), float(np.mean(backward <= tau))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def iou_from_volumes(prediction: float, reference: float, intersection: float) -> float:
    values = np.array([prediction, reference, intersection])
    if not np.isfinite(values).all() or min(prediction, reference) <= 0 or intersection < 0:
        raise ValueError("Объёмы должны быть конечными, объёмы тел > 0, пересечение >= 0")
    if intersection > min(prediction, reference):
        raise ValueError("Пересечение больше одного из тел: ошибка ядра или единиц")
    return intersection / (prediction + reference - intersection)


def load_mesh(path: Path) -> trimesh.Trimesh:
    if path.suffix.lower() != ".stl":
        raise ValueError("Ожидается STL в миллиметрах")
    return trimesh.load_mesh(path, process=True)


def surface_distances(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    return np.concatenate([mesh.nearest.on_surface(batch)[1] for batch in np.array_split(points, 32)])


def compare(prediction: trimesh.Trimesh, reference: trimesh.Trimesh, settings: EvaluationSettings) -> dict:
    if not reference.is_volume:
        raise ValueError("Эталон должен быть замкнутой ориентированной объёмной сеткой")
    tau = settings.tau_relative * float(np.linalg.norm(reference.extents))
    if not prediction.is_volume:
        return {"status": "invalid_prediction", "iou": 0.0, "f_score": 0.0, "tau_mm": tau}
    intersection = trimesh.boolean.intersection([prediction, reference], engine="manifold", check_volume=True)
    volumes = [
        float(prediction.volume),
        float(reference.volume),
        0.0 if intersection.is_empty else float(intersection.volume),
    ]
    tolerance = settings.intersection_roundoff_relative * min(volumes[:2])
    if not -tolerance <= volumes[2] <= min(volumes[:2]) + tolerance:
        raise ValueError("Некорректный объём пересечения")
    intersection_volume = float(np.clip(volumes[2], 0, min(volumes[:2])))
    pred_points, _ = trimesh.sample.sample_surface(
        prediction, settings.surface_samples, seed=settings.seed + 100
    )
    ref_points, _ = trimesh.sample.sample_surface(
        reference, settings.surface_samples, seed=settings.seed + 101
    )
    return {
        "status": "ok",
        "iou": iou_from_volumes(volumes[0], volumes[1], intersection_volume),
        "f_score": f_score(
            surface_distances(reference, pred_points), surface_distances(prediction, ref_points), tau
        ),
        "tau_mm": tau,
        "volumes_mm3": {
            "prediction": volumes[0],
            "reference": volumes[1],
            "intersection": intersection_volume,
            "intersection_roundoff_correction": intersection_volume - volumes[2],
        },
    }


def evaluate(
    model_path: Path, reference_path: Path, settings: EvaluationSettings, generation: dict | None = None
) -> dict:
    record = {
        "protocol": "mesh-fixed-pose-v1",
        "status": "error",
        "iou": None,
        "f_score": None,
        "parameters": {
            **settings.model_dump(),
            "units": "mm",
            "alignment": "identity",
            "distance": "point_to_triangle",
            "intersection": "manifold",
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "trimesh", "rtree", "manifold3d")
        },
        "evaluator_sha256": sha256(Path(__file__)),
    }
    stage = "проверка эталонной сетки"
    try:
        reference_hash = sha256(reference_path)
        manifest_path = reference_path.parent.parent / "manifest.json"
        if reference_path.parent.name == "target" and manifest_path.is_file():
            manifest = read_json(manifest_path)
            if reference_hash != manifest["files"]["target/" + reference_path.name]:
                raise ValueError("SHA256 эталона не совпал с описью")
        record["reference"] = {"name": reference_path.name, "sha256": reference_hash}
        reference = load_mesh(reference_path)
        if not reference.is_volume:
            raise ValueError("Некорректный эталон")
        stage = "проверка результата генерации"
        if generation and generation["status"] != "success":
            if generation["status"] == "failed" and generation.get("failure_kind") == "generation":
                record.update(status="generation_failed", iou=0.0, f_score=0.0)
                return record
            raise ValueError("Запуск прерван или завершился ошибкой инфраструктуры")
        actual_hash = sha256(model_path)
        expected = (generation or {}).get("cad", {}).get("files", {}).get(model_path.name)
        if expected is not None and actual_hash != expected:
            raise ValueError("SHA256 результата не совпал с журналом построения")
        record["prediction"] = {"name": model_path.name, "sha256": actual_hash}
        stage = "расчёт IoU и F-score"
        record.update(compare(load_mesh(model_path), reference, settings))
    except Exception as error:  # noqa: BLE001 — ошибка измерителя остаётся отсутствующим значением.
        details = error_details(error, stage)
        record.update(
            status="error",
            iou=None,
            f_score=None,
            error=details,
            reason=f"{details['type']}: {details['message']}",
        )
    return record


def check_data(data: Path) -> None:
    manifest = read_json(data / "manifest.json")
    names = [item["pair_id"] for item in manifest["items"]]
    if len(names) != len(set(names)):
        raise ValueError("Деталь повторяется в описи")
    for path, expected in manifest["files"].items():
        if sha256(data / path) != expected:
            raise ValueError(f"SHA256 не совпал: {path}")
    for item in manifest["items"]:
        target = item["target_mesh"]
        if sha256(data / target["path"]) != target["sha256"]:
            raise ValueError("Изменён эталон")
        matrix = np.asarray(target["source_to_target"])
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3)) or not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError("Оси должны быть ортогональными, без масштаба и отражения")
        original = load_mesh(data / target["source_mesh"])
        original.apply_transform(matrix)
        if original.export(file_type="stl") != (data / target["path"]).read_bytes():
            raise ValueError("Эталон не соответствует исходной сетке и преобразованию")
        mesh = load_mesh(data / target["path"])
        if not mesh.is_volume or abs(mesh.bounds[0, 2]) > 1e-5:
            raise ValueError("Некорректное тело или нижний торец не на Z=0")


def cached_result(run: Run, settings: EvaluationSettings) -> dict | None:
    """Среднее использует только оценки текущих файлов с одинаковыми параметрами."""
    try:
        record = read_json(run.artifacts / "evaluation.json")
        if any(record["parameters"].get(key) != value for key, value in settings.model_dump().items()):
            return None
        reference = run.root / "data/target" / f"{run.pair_id}.stl"
        if sha256(reference) != record["reference"]["sha256"]:
            return None
        if record.get("generation_sha256"):
            if sha256(run.artifacts / "run.json") != record["generation_sha256"]:
                return None
        if record["status"] == "generation_failed":
            if not record.get("generation_sha256"):
                return None
        elif sha256(run.directory / "model.stl") != record["prediction"]["sha256"]:
            return None
        for key in ("iou", "f_score"):
            if record[key] is None or not math.isfinite(record[key]) or not 0 <= record[key] <= 1:
                return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_report(root: Path, run_name: str, settings: EvaluationSettings) -> dict:
    items = read_json(root / "data/manifest.json")["items"]
    if len({item["pair_id"] for item in items}) != len(items):
        raise ValueError("Деталь повторяется в описи")
    scores = {
        item["pair_id"]: cached_result(Run(item["pair_id"], run_name, root), settings) for item in items
    }
    validation = [item["pair_id"] for item in items if item["split"] == "val"]
    measured = [scores[pair] for pair in validation if scores[pair] is not None]
    summary = {"measured": len(measured), "expected": len(validation), "iou": None, "f_score": None}
    if validation and len(measured) == len(validation):
        for key in ("iou", "f_score"):
            summary[key] = math.fsum(score[key] for score in measured) / len(validation)

    def cells(score: dict | None) -> str:
        return " | ".join(
            "—" if score is None or score[key] is None else f"{score[key]:.6f}" for key in ("iou", "f_score")
        )

    lines = ["| Деталь | IoU | F-score |", "|---|---:|---:|"]
    for item in items:
        label = item["pair_id"]
        lines.append(f"| {label} | {cells(scores[item['pair_id']])} |")
    lines.append(f"| **Среднее val ({len(measured)}/{len(validation)})** | {cells(summary)} |")
    atomic(root / "results" / run_name / "metrics.md", ("\n".join(lines) + "\n").encode())
    return summary


def evaluate_run(run: Run, settings: EvaluationSettings) -> dict:
    logfile = run.logfile if run.directory.is_dir() else None
    log("ОЦЕНКА", f"{run.name}: считаю IoU и F-score.", logfile)
    journal = run.artifacts / "run.json"
    try:
        generation = read_json(journal) if journal.is_file() else None
        record = evaluate(
            run.directory / "model.stl", run.root / "data/target" / f"{run.pair_id}.stl", settings, generation
        )
        if journal.is_file():
            record["generation_sha256"] = sha256(journal)
    except Exception as error:  # noqa: BLE001 — сохраняем также ошибку чтения журнала.
        details = error_details(error, "чтение журнала генерации")
        record = {
            "status": "error",
            "iou": None,
            "f_score": None,
            "reason": details["message"],
            "error": details,
        }
    record.update(pair_id=run.pair_id, run_name=run.run_name)
    write_json(run.artifacts / "evaluation.json", record)
    if record["status"] == "error":
        log_error(record["error"], run.artifacts / "evaluation-error.log", logfile)
    else:
        log("ГОТОВО", f"IoU {record['iou']:.6f} · F-score {record['f_score']:.6f}", logfile)
        if record["status"] != "ok":
            log("ВНИМАНИЕ", "Построение не удалось или сетка некорректна: метрики равны нулю.", logfile)
    return record


def evaluate_selected(
    selector: str | None, settings: EvaluationSettings, root: Path = ROOT, run_name: str = "baseline"
) -> int:
    records = [evaluate_run(run, settings) for run in select_runs(selector, root, run_name)]
    summary = write_report(root, run_name, settings)
    if summary["iou"] is not None:
        log("СРЕДНЕЕ", f"val: IoU {summary['iou']:.6f} · F-score {summary['f_score']:.6f}")
    else:
        log("ВНИМАНИЕ", f"Среднее val не вычислено: оценено {summary['measured']} из {summary['expected']}.")
    log("РЕЗУЛЬТАТ", f"Таблица: results/{run_name}/metrics.md")
    return 1 if any(record["status"] != "ok" for record in records) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Две метрики STL, независимо от агента")
    parser.add_argument("selector", nargs="?", help="Деталь или папка прогона; без аргумента — все")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    args = parser.parse_args()
    stage = "чтение конфигурации оценки"
    try:
        config = load_config(args.config)
        stage = "оценка и сохранение метрик"
        return evaluate_selected(args.selector, config.evaluation, run_name=config.name)
    except Exception as error:  # noqa: BLE001 — сохраняем ошибки команды оценки.
        log_error(error_details(error, stage), ROOT / "artifacts/evaluation-command-error.log")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
