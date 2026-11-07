"""Контракт с вычислителем КОМПАС: преобразование и хеши входят в ключ запроса."""

from typing import Any

import numpy as np

from eval.geometry import PROTOCOL_VERSION
from eval.geometry.alignment import validate_rigid
from eval.geometry.io import digest
from eval.geometry.models import BooleanResult, Facts


def solid_request(
    case_id: str,
    pose: str,
    prediction_hash: str | None,
    reference_hash: str | None,
    pred_mesh_hash: str,
    ref_mesh_hash: str,
    transform: np.ndarray,
    prediction_units: str = "mm",
    reference_units: str = "mm",
) -> dict[str, Any]:
    validate_rigid(transform)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": case_id,
        "pose": pose,
        "units": "mm",
        "prediction_native_sha256": prediction_hash,
        "reference_native_sha256": reference_hash,
        "prediction_mesh_sha256": pred_mesh_hash,
        "reference_mesh_sha256": ref_mesh_hash,
        "prediction_mesh_units": prediction_units,
        "reference_mesh_units": reference_units,
        "prediction_to_reference": transform.tolist(),
    }
    return {"request_sha256": digest(payload), **payload}


def iou_from_volumes(prediction: float, reference: float, intersection: float) -> dict[str, float]:
    values = np.array([prediction, reference, intersection])
    if not np.isfinite(values).all() or min(prediction, reference) <= 0 or intersection < 0:
        raise ValueError("Объёмы должны быть конечными, объёмы тел > 0, пересечение >= 0")
    if intersection > min(prediction, reference):
        raise ValueError("Пересечение больше одного из тел: ошибка ядра или единиц")
    union = prediction + reference - intersection
    return {
        "iou": intersection / union,
        "intersection_mm3": intersection,
        "union_mm3": union,
        "symmetric_difference_mm3": prediction + reference - 2 * intersection,
        "extra_material_mm3": prediction - intersection,
        "missing_material_mm3": reference - intersection,
    }


def evaluate_solid(
    request: dict[str, Any],
    results: dict[str, BooleanResult],
    prediction: Facts | None,
    reference: Facts | None,
    validation: bool,
) -> dict[str, Any]:
    result = results.get(request["request_sha256"])
    if result is None:
        return {
            "status": "not_computed",
            "value": None,
            "reason": "Нужен результат булевого пересечения КОМПАС для этого запроса",
        }
    if result.backend != "kompas_v25" and not validation:
        return {"status": "error", "value": None, "reason": "Аналитический стенд не является оценкой CAD"}
    if result.status == "error":
        return {"status": "error", "value": None, "reason": result.reason}
    if not validation and (not request["prediction_native_sha256"] or not request["reference_native_sha256"]):
        return {"status": "error", "value": None, "reason": "Для IoU нужны хеши исходных CAD-документов"}
    assert result.prediction_volume_mm3 is not None and result.reference_volume_mm3 is not None
    assert result.intersection_volume_mm3 is not None
    for facts, volume in (
        (prediction, result.prediction_volume_mm3),
        (reference, result.reference_volume_mm3),
    ):
        if (
            facts is not None
            and facts.volume_mm3 is not None
            and not np.isclose(facts.volume_mm3, volume, rtol=1e-6, atol=1e-8)
        ):
            return {
                "status": "error",
                "value": None,
                "reason": "Объём булева запроса не совпал со снимком R2",
            }
    try:
        metrics = iou_from_volumes(
            result.prediction_volume_mm3, result.reference_volume_mm3, result.intersection_volume_mm3
        )
    except ValueError as error:
        return {"status": "error", "value": None, "reason": str(error)}
    return {
        "status": "ok",
        "value": metrics["iou"],
        "backend": result.backend,
        "backend_version": result.backend_version,
        **metrics,
    }
