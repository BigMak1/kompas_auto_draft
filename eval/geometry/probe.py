"""Живая проверка экспортёра и пересечения на заранее сохранённых контрольных кубах.

В <run>/inputs: prediction.m3d — куб [0,10]^3, reference.m3d — тот же куб со сдвигом X=5.
Зачем: проверить единицы МЦХ, поворот/перенос и пустое пересечение через настоящий KsAPI.
Запуск — таск eval-geometry-kompas-smoke; исходные файлы открываются только копиями.
"""

import argparse
from pathlib import Path

import numpy as np

from eval.geometry.backend import WorkerClient, settings_from_path
from eval.geometry.io import atomic_json, create_artifact_directory, sha256
from eval.geometry.models import BooleanResult, ExportConfig
from eval.geometry.normalize import normalize_export
from eval.geometry.solid import iou_from_volumes, solid_request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--backend", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    output = create_artifact_directory(root, "probe")
    client = WorkerClient(settings_from_path(args.backend), output / "worker")
    report = {"purpose": "metric_validation", "health": client.health(), "checks": []}
    facts = {}
    sources = {role: root / "inputs" / f"{role}.m3d" for role in ("prediction", "reference")}
    for role, source in sources.items():
        job, _ = client.invoke("export", {"profile": ExportConfig().model_dump()}, {"model": source})
        facts[role] = normalize_export(job)
        item = facts[role]
        if not np.isclose(item.volume_mm3, 1000) or not np.isclose(item.area_mm2, 600):
            raise ValueError("Контрольные тела должны быть кубами 10×10×10 мм")
        center = [5, 5, 5] if role == "prediction" else [10, 5, 5]
        if not np.allclose(item.center_mm, center, atol=1e-6):
            raise ValueError("Неверные координаты контрольного куба")
        if not np.allclose(item.principal_moments, item.mass_g * 100 / 6, rtol=1e-6):
            raise ValueError("Единицы главных моментов не согласованы с г·мм²")
        if item.topology != {"faces": 6, "edges": 12, "vertices": 8}:
            raise ValueError(f"Неверная топология контрольного куба: {item.topology}")
        report["checks"].append({"export": role, "status": "ok", "job": str(job)})
        atomic_json(output / "checks.json", report)
    rotate = np.array([[0.0, -1, 0, 15], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    translate = np.eye(4)
    translate[0, 3] = 30
    for name, matrix, bounds, expected in (
        ("overlap", np.eye(4), [0, 0, 0, 10, 10, 10], 500),
        ("rotate_and_translate", rotate, [5, 0, 0, 15, 10, 10], 1000),
        ("disjoint", translate, [30, 0, 0, 40, 10, 10], 0),
    ):
        request = solid_request(
            "control-cubes",
            name,
            sha256(sources["prediction"]),
            sha256(sources["reference"]),
            facts["prediction"].mesh_sha256,
            facts["reference"].mesh_sha256,
            matrix,
        )
        job, response = client.invoke(
            "intersect",
            {
                "solid_request": request,
                "prediction_bounds_mm": bounds,
                "bbox_tolerance_mm": 0.02,
            },
            sources,
        )
        result = BooleanResult.model_validate(response["result"])
        if result.status != "ok" or not np.isclose(result.intersection_volume_mm3, expected, atol=1e-6):
            raise ValueError(f"Контроль {name} не пройден: {result}")
        values = iou_from_volumes(
            result.prediction_volume_mm3,
            result.reference_volume_mm3,
            result.intersection_volume_mm3,
        )
        report["checks"].append({"case": name, "status": "ok", "job": str(job), **values})
        atomic_json(output / "checks.json", report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
