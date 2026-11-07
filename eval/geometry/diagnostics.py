"""Диагностика сетки и сравнение заранее сопоставленных измерений ядра."""

from typing import Any

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from eval.geometry.models import Config, Facts, Requirement
from eval.geometry.progress import ProgressCallback, no_progress


def mesh_diagnostics(mesh: trimesh.Trimesh, progress: ProgressCallback = no_progress) -> dict[str, Any]:
    progress("Связность, разрывы, ориентация и Flux Enclosure Error")
    edges = mesh.edges_unique
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(mesh.vertices), len(mesh.vertices))
    )
    segments = int(connected_components(graph, directed=False, return_labels=False))
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(edges))
    boundary = edges[counts == 1]
    dangling = np.linalg.norm(mesh.vertices[boundary[:, 0]] - mesh.vertices[boundary[:, 1]], axis=1).sum()
    flux_vector = np.sum(mesh.face_normals * mesh.area_faces[:, None], axis=0)
    result: dict[str, Any] = {
        "connected_segments": segments,
        "dangling_edge_length_mm": float(dangling),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "flux_enclosure_error_mm2": float(abs(flux_vector.sum())),
        "flux_vector_norm_mm2": float(np.linalg.norm(flux_vector)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "degenerate_faces": int(np.sum(mesh.area_faces <= 0)),
    }
    try:
        progress("Self-Intersection Ratio: проверка самопересечений")
        from igl.copyleft.cgal import remesh_self_intersections

        # Только обнаружение: возвращаемую геометрию не используем. CGAL корректно
        # отличает общую вершину/ребро от пересечения соседних треугольников.
        _, _, pairs, _, _ = remesh_self_intersections(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            detect_only=True,
            first_only=False,
        )
        result["self_intersection_ratio"] = {
            "status": "ok",
            "value": len(np.unique(pairs)) / len(mesh.faces),
            "intersecting_face_pairs": len(pairs),
            "backend": "libigl.copyleft.cgal.detect_only",
        }
    except (ImportError, RuntimeError, ValueError) as error:
        result["self_intersection_ratio"] = {"status": "not_computed", "value": None, "reason": str(error)}
    return result


def match_dimensions(prediction: list[float], reference: list[float], tolerance: float) -> dict[str, Any]:
    """Максимум совпадений с допуском, затем минимум ошибки; повторы не теряются."""
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Неверный допуск")
    matched = 0
    if prediction and reference:
        errors = np.abs(np.array(prediction)[:, None] - np.array(reference)[None, :])
        allowed = errors <= tolerance
        penalty = min(len(prediction), len(reference)) + 1
        costs = np.where(allowed, errors / max(tolerance, 1e-12), penalty)
        rows, cols = linear_sum_assignment(costs)
        matched = int(allowed[rows, cols].sum())
    return {
        "matched": matched,
        "prediction_count": len(prediction),
        "reference_count": len(reference),
        "precision": matched / len(prediction) if prediction else (1.0 if not reference else 0.0),
        "recall": matched / len(reference) if reference else 1.0,
    }


def requirement_metrics(requirements: list[Requirement], measurements: dict[str, float]) -> dict[str, Any]:
    rows = []
    for requirement in requirements:
        value = measurements.get(requirement.id)
        error = abs(value - requirement.target) if value is not None else None
        rows.append(
            {
                **requirement.model_dump(),
                "measured": value,
                "absolute_error": error,
                "passed": error is not None and error <= requirement.tolerance,
            }
        )
    dimensions = [row for row in rows if row["kind"] == "dimension"]
    constraints = [row for row in rows if row["kind"] == "constraint"]
    result = {
        "checks": rows,
        "specification_compliance": sum(r["passed"] for r in rows) / len(rows) if rows else None,
        "dimension_compliance": sum(r["passed"] for r in dimensions) / len(dimensions)
        if dimensions
        else None,
        "g_csr_project": sum(r["passed"] for r in constraints) / len(constraints) if constraints else None,
    }
    for unit in ("mm", "deg", "ratio"):
        relevant = [row for row in dimensions if row["unit"] == unit]
        measured = [row["absolute_error"] for row in relevant if row["absolute_error"] is not None]
        result[f"dimension_absolute_error_{unit}_mean"] = float(np.mean(measured)) if measured else None
        result[f"dimension_{unit}_measured_count"] = len(measured)
        result[f"dimension_{unit}_required_count"] = len(relevant)
    return result


def compare_facts(
    prediction: Facts | None,
    reference: Facts | None,
    transform: np.ndarray,
    config: Config,
    progress: ProgressCallback = no_progress,
) -> dict[str, Any]:
    if prediction is None or reference is None:
        progress("Инженерные метрики недоступны: нет снимка измерений")
        return {"status": "not_computed", "reason": "Нет снимка R2"}
    progress("МЦХ: объём, площадь, масса, центр и моменты инерции")
    result: dict[str, Any] = {"status": "ok"}
    for field in ("volume_mm3", "area_mm2", "mass_g"):
        pred, ref = getattr(prediction, field), getattr(reference, field)
        if field == "mass_g" and (not reference.material or prediction.material != reference.material):
            result[f"{field}_relative_error"] = None
        else:
            result[f"{field}_relative_error"] = abs(pred - ref) / ref if pred is not None and ref else None
    if prediction.center_mm is not None and reference.center_mm is not None:
        center = transform[:3, :3] @ np.array(prediction.center_mm) + transform[:3, 3]
        result["center_error_mm"] = float(np.linalg.norm(center - reference.center_mm))
    if (
        prediction.principal_moments is not None
        and reference.principal_moments is not None
        and reference.inertia_unit
        and prediction.inertia_unit == reference.inertia_unit
        and (
            reference.inertia_unit == "mm5"
            or (reference.material and prediction.material == reference.material)
        )
    ):
        pred, ref = np.sort(prediction.principal_moments), np.sort(reference.principal_moments)
        result["principal_moments_relative_error"] = (np.abs(pred - ref) / ref).tolist()
    else:
        result["principal_moments_relative_error"] = None
    progress("Топология: сравнение числа граней, рёбер и вершин")
    result["topology_delta"] = {
        key: prediction.topology[key] - value if key in prediction.topology else None
        for key, value in reference.topology.items()
    }
    result["dimension_sets"] = {}
    groups = sorted(prediction.dimensions_mm.keys() | reference.dimensions_mm.keys())
    names = {"diameter": "диаметры", "radius": "радиусы", "linear": "линейные", "thread": "резьбы"}
    for index, key in enumerate(groups, start=1):
        pred_values = prediction.dimensions_mm.get(key, [])
        ref_values = reference.dimensions_mm.get(key, [])
        progress(
            f"Размеры {index}/{len(groups)} · {names.get(key, key)}: "
            f"{len(pred_values)} ↔ {len(ref_values)} значений"
        )
        result["dimension_sets"][key] = match_dimensions(
            pred_values, ref_values, config.dimension_tolerance_mm
        )
    return result
