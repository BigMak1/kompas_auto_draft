"""Независимые метрики геометрии STL: IoU и F-score@τ, без OpenAI SDK, CadQuery и MLflow."""

import importlib.metadata
from pathlib import Path

import numpy as np
import trimesh

from src.core.config import EvaluationSettings
from src.core.utils import error_details, read_json, sha256


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
    model_path: Path,
    reference_path: Path,
    settings: EvaluationSettings,
    generation: dict | None = None,
    *,
    expected_reference_sha256: str | None = None,
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
        if expected_reference_sha256 is not None:
            if reference_hash != expected_reference_sha256:
                raise ValueError("SHA256 эталона не совпал со снимком эксперимента")
        elif reference_path.parent.name == "target" and manifest_path.is_file():
            manifest = read_json(manifest_path)
            targets = [
                item["target_mesh"]
                for item in manifest["items"]
                if item["target_mesh"]["path"] == "target/" + reference_path.name
            ]
            if len(targets) != 1:
                raise ValueError("Эталон должен встречаться в манифесте датасета ровно один раз")
            if reference_hash != targets[0]["sha256"]:
                raise ValueError("SHA256 эталона не совпал с манифестом датасета")
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
