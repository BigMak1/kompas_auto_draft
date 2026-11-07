"""Проверка чтения STL и устойчивости диагностики к повороту, без КОМПАС.

Запуск: uv run poe eval-geometry-mesh-probe <run>. Исходные файлы не изменяются;
сохраняется новый отчёт с хешами, версией кода и результатами каждого положения.
"""

import argparse
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from eval.geometry.diagnostics import mesh_diagnostics
from eval.geometry.io import atomic_json, atomic_text, create_artifact_directory, sha256
from eval.geometry.pipeline import load_evaluation
from eval.geometry.runner import SCALE_MM, load_model


def inspect_meshes(root: Path, rotations: int = 8) -> Path:
    root = root.resolve()
    evaluation = load_evaluation(root)
    output = create_artifact_directory(root / "evaluation", "mesh-probe")
    records: list[dict[str, Any]] = []
    for case in evaluation.cases:
        for role in ("reference", "prediction"):
            model = getattr(case, role)
            if model is None:
                continue
            mesh, facts, provenance = load_model(model, root, evaluation.purpose == "metric_validation")
            raw = trimesh.load_mesh(root / model.mesh, process=False)
            vertices, inverse = np.unique(raw.vertices, axis=0, return_inverse=True)
            exact = trimesh.Trimesh(
                vertices=vertices * SCALE_MM[model.units], faces=inverse[raw.faces], process=False
            )
            poses = []
            for seed in range(rotations + 1):
                transform = np.eye(4)
                if seed:
                    transform[:3, :3] = Rotation.random(random_state=seed).as_matrix()
                    transform[:3, 3] = [13.1, -21.2, 36.3]
                moved = mesh.copy()
                moved.apply_transform(transform)
                poses.append({"transform": transform.tolist(), "diagnostics": mesh_diagnostics(moved)})
            records.append(
                {
                    "case_id": case.id,
                    "role": role,
                    "inputs": provenance,
                    "kernel": facts.model_dump() if facts else None,
                    "exact_vertices_diagnostics": mesh_diagnostics(exact),
                    "normalized_poses": poses,
                    "triangle_count": len(mesh.faces),
                    "area_before_mm2": exact.area,
                    "area_after_mm2": mesh.area,
                    "volume_before_mm3": exact.volume,
                    "volume_after_mm3": mesh.volume,
                    "source_file_unchanged": sha256(root / model.mesh) == provenance["mesh"],
                }
            )
    atomic_json(
        output / "diagnostics.json",
        {
            "models": records,
            "versions": {name: version(name) for name in ("numpy", "scipy", "trimesh", "libigl")},
            "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        },
    )
    atomic_text(
        output / "notes.md",
        "# Диагностика сеток\n\n## Кратко\n\n"
        "Проверены исходные файлы, нормализация округления и повороты с переносом. "
        "Модели и STL не перезаписывались. Результаты: [diagnostics.json](diagnostics.json).\n\n"
        f"Команда: `uv run poe eval-geometry-mesh-probe {root} --rotations {rotations}`.\n",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--rotations", type=int, default=8)
    args = parser.parse_args()
    if args.rotations < 1:
        parser.error("Нужен хотя бы один поворот")
    print(inspect_meshes(args.run, args.rotations))


if __name__ == "__main__":
    main()
