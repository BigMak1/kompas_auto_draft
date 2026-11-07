"""Синтетический стенд; не читает корпус и не заменяет булево ядро КОМПАС."""

import argparse
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from eval.geometry import PROTOCOL_VERSION
from eval.geometry.io import atomic_json, atomic_text, create_artifact_directory, read_json, sha256
from eval.geometry.models import Evaluation
from eval.geometry.progress import ProgressCallback, no_progress, progress_display
from eval.geometry.runner import finalize_solid_results, run_evaluation
from eval.geometry.terminal import add_output_arguments, print_summary


def write_box(root: Path, name: str, extents: tuple[float, float, float], transform: np.ndarray) -> dict:
    """МЦХ прямоугольного параллелепипеда из аналитических формул, только для стенда."""
    box = trimesh.creation.box(extents=extents, transform=transform)
    # ASCII PLY сохраняет геометрию независимо от внешнего ПО; запись атомарна.
    mesh_path = root / f"{name}.ply"
    atomic_text(mesh_path, box.export(file_type="ply", encoding="ascii").decode())
    volume = float(np.prod(extents))
    moments = volume / 12 * (np.sum(np.array(extents) ** 2) - np.array(extents) ** 2)
    facts = {
        "source": "analytic_fixture",
        "source_version": "box-formulas-v1",
        "mesh_sha256": sha256(mesh_path),
        "mesh_units": "mm",
        "solid_valid": True,
        "body_count": 1,
        "volume_mm3": volume,
        "area_mm2": float(2 * (extents[0] * extents[1] + extents[0] * extents[2] + extents[1] * extents[2])),
        "center_mm": transform[:3, 3].tolist(),
        "principal_axes": transform[:3, :3].tolist(),
        "principal_moments": moments.tolist(),
        "inertia_unit": "mm5",
        "measurements": {"width": extents[0]},
    }
    atomic_json(root / f"{name}.facts.json", facts)
    return {"mesh": mesh_path.name, "units": "mm", "facts": f"{name}.facts.json"}


def convex_fixture_intersection(prediction: trimesh.Trimesh, reference: trimesh.Trimesh) -> float:
    """Независимый тестовый оракул для выпуклых коробок, НЕ production IoU."""
    planes = np.unique(
        np.round(
            np.vstack((ConvexHull(prediction.vertices).equations, ConvexHull(reference.vertices).equations)),
            12,
        ),
        axis=0,
    )
    points = []
    for triple in combinations(planes, 3):
        rows = np.asarray(triple)
        if abs(np.linalg.det(rows[:, :3])) < 1e-9:
            continue
        point = np.linalg.solve(rows[:, :3], -rows[:, 3])
        if np.all(planes[:, :3] @ point + planes[:, 3] <= 1e-8):
            points.append(point)
    if len(points) < 4:
        return 0.0
    points = np.unique(np.round(points, 10), axis=0)
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-9) < 3:
        return 0.0
    return float(ConvexHull(points).volume)


def create_smoke(root: Path | None = None, progress: ProgressCallback = no_progress) -> Path:
    started = time.perf_counter()
    if root is None:
        root = create_artifact_directory(Path("runs/m2"), "m2-synthetic").resolve()
    else:
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=False)
    identity = np.eye(4)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [33, 51, 17], degrees=True).as_matrix()
    pose[:3, 3] = [20, -11, 7]
    reference = write_box(root, "reference", (2, 3, 5), identity)
    rotated = write_box(root, "rotated", (2, 3, 5), pose)
    wrong_size = write_box(root, "wrong_size", (2.4, 3, 5), pose)
    requirement = {"id": "width", "kind": "dimension", "target": 2, "tolerance": 0.01, "unit": "mm"}
    spec = {
        "purpose": "metric_validation",
        "config": {
            "surface_samples": 512,
            "registration_samples": 256,
            "edge_samples": 256,
            "icp_iterations": 15,
            "icp_candidates": 4,
        },
        "cases": [
            {
                "id": "rigid_box",
                "family": "synthetic_box",
                "reference": reference,
                "prediction": rotated,
                "requirements": [requirement],
                "tags": ["rigid_pose"],
            },
            {
                "id": "wrong_width",
                "family": "synthetic_box",
                "reference": reference,
                "prediction": wrong_size,
                "requirements": [requirement],
                "tags": ["dimension_error"],
            },
            {
                "id": "failed_generation",
                "family": "synthetic_box",
                "reference": reference,
                "generation_status": "failed",
                "requirements": [requirement],
            },
        ],
    }
    atomic_json(root / "evaluation.json", spec)
    atomic_json(
        root / "raw" / "fixture-definition.json",
        {
            "reference_extents_mm": [2, 3, 5],
            "wrong_width_extents_mm": [2.4, 3, 5],
            "prediction_pose": pose.tolist(),
            "config": spec["config"],
            "seed": 20261107,
        },
    )
    first = run_evaluation(root, progress=progress)
    requests = read_json(first / "raw" / "solid_requests.json")
    by_id = {case["id"]: case for case in spec["cases"]}
    boolean_results = []
    for request in requests:
        case = by_id[request["case_id"]]
        pred = trimesh.load_mesh(root / case["prediction"]["mesh"])
        ref = trimesh.load_mesh(root / case["reference"]["mesh"])
        pred.apply_transform(request["prediction_to_reference"])
        pred_volume = read_json(root / case["prediction"]["facts"])["volume_mm3"]
        ref_volume = read_json(root / case["reference"]["facts"])["volume_mm3"]
        # Округление ASCII PLY учитывается только в оракуле стенда, не в production IoU.
        intersection = min(convex_fixture_intersection(pred, ref), pred_volume, ref_volume)
        boolean_results.append(
            {
                "request_sha256": request["request_sha256"],
                "backend": "analytic_fixture",
                "backend_version": "convex-box-halfspaces-v1",
                "status": "ok",
                "prediction_volume_mm3": pred_volume,
                "reference_volume_mm3": ref_volume,
                "intersection_volume_mm3": intersection,
            }
        )
    atomic_json(root / "boolean-results.json", boolean_results)
    spec["boolean_results"] = "boolean-results.json"
    atomic_json(root / "evaluation.json", spec)
    final = finalize_solid_results(root, first, Evaluation.model_validate(spec), progress)
    atomic_json(
        root / "manifest.json",
        {
            "run_id": root.name,
            "track": "m2",
            "purpose": "metric_validation",
            "date": datetime.now(UTC).isoformat(),
            "inputs": [
                {"path": str(path.relative_to(root)), "sha256": sha256(path)}
                for path in sorted(root.iterdir())
                if path.is_file()
            ],
            "settings": spec["config"],
            "command": "uv run poe eval-geometry-smoke",
            "result": str(final.relative_to(root)),
            "model": "not_applicable",
            "prompt_version": None,
            "schema_version": PROTOCOL_VERSION,
            "cost": "See evaluation/*/manifest.json",
            "source_sha256": sha256(Path(__file__)),
        },
    )
    atomic_text(
        root / "notes.md",
        "# Синтетическая проверка метрик\n\n## Кратко\n\n"
        "Проверены поворот с переносом, ошибка ширины и неудачная генерация. "
        "IoU получен аналитическим тестовым оракулом коробок, не КОМПАС. "
        "Это не результат реконструкции по чертежу.\n\n"
        f"Итог: [{final.name}/report.md]({final.relative_to(root)}/report.md). "
        "Первичная оценка без булевых результатов тоже сохранена.\n",
    )
    manifest = read_json(final / "manifest.json")
    manifest["cost"]["total_seconds"] = time.perf_counter() - started
    atomic_json(final / "manifest.json", manifest)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    add_output_arguments(parser)
    args = parser.parse_args()
    with progress_display(quiet=args.quiet, no_color=args.no_color) as progress:
        result = create_smoke(args.output, progress)
    print_summary(
        read_json(result / "metrics.json"),
        result,
        all_metrics=args.all_metrics,
        no_color=args.no_color,
        quiet=args.quiet,
        elapsed_seconds=read_json(result / "manifest.json")["cost"]["total_seconds"],
    )


if __name__ == "__main__":
    main()
