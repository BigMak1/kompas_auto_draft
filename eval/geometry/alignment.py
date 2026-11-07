"""Регистрация SE(3): оси ядра → кандидаты → rigid ICP → одна матрица для всех метрик."""

from itertools import permutations, product
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from eval.geometry.models import Config, Facts
from eval.geometry.progress import ProgressCallback, no_progress

Array = NDArray[np.float64]


def validate_rigid(matrix: Array) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Ожидается конечная матрица 4×4")
    rotation = matrix[:3, :3]
    if not (
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        and np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
        and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
    ):
        raise ValueError("Разрешены только поворот и перенос: масштаб и отражения запрещены")


def rigid_candidates(prediction: Facts, reference: Facts, config: Config) -> list[Array]:
    for facts in (prediction, reference):
        if facts.center_mm is None or facts.principal_axes is None or facts.principal_moments is None:
            raise ValueError("Для rigid нужны center_mm, principal_axes и principal_moments из R2")
    pred_axes = np.array(prediction.principal_axes)
    ref_axes = np.array(reference.principal_axes)
    center_pred, center_ref = np.array(prediction.center_mm), np.array(reference.center_mm)
    spins = [np.eye(3)]
    # При близких моментах собственные векторы внутри подпространства неустойчивы.
    for i, j in ((0, 1), (0, 2), (1, 2)):
        values = np.array(reference.principal_moments)
        pred_values = np.sort(prediction.principal_moments)
        pred_degenerate = np.min(np.diff(pred_values) / pred_values[1:]) < config.inertia_relative_gap
        if (
            abs(values[i] - values[j]) / max(values[i], values[j]) < config.inertia_relative_gap
            or pred_degenerate
        ):
            axis = ref_axes[:, 3 - i - j]
            spins.extend(
                Rotation.from_rotvec(axis * angle).as_matrix()
                for angle in np.linspace(0, 2 * np.pi, config.symmetry_steps, endpoint=False)[1:]
            )
    candidates = []
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            mapping = np.eye(3)[:, order] @ np.diag(signs)
            base = ref_axes @ mapping @ pred_axes.T
            if np.linalg.det(base) < 0:
                continue
            for spin in spins:
                matrix = np.eye(4)
                matrix[:3, :3] = spin @ base
                matrix[:3, 3] = center_ref - matrix[:3, :3] @ center_pred
                validate_rigid(matrix)
                candidates.append(matrix)
    return candidates


def align(
    prediction: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    pred_facts: Facts | None,
    ref_facts: Facts | None,
    config: Config,
    progress: ProgressCallback = no_progress,
) -> tuple[Array, dict[str, Any]]:
    if config.alignment == "fixed":
        return np.eye(4), {"method": "fixed", "candidates": 1}
    if pred_facts is None or ref_facts is None:
        raise ValueError("Rigid-совмещение требует снимков R2 обеих моделей")
    progress("Совмещение: подготовка возможных положений")
    candidates = rigid_candidates(pred_facts, ref_facts, config)
    points, _ = trimesh.sample.sample_surface(prediction, config.registration_samples, seed=config.seed)
    ref_points, _ = trimesh.sample.sample_surface(
        reference, config.registration_samples, seed=config.seed + 1
    )
    ref_tree = cKDTree(ref_points)

    def coarse_cost(matrix: Array) -> float:
        moved = trimesh.transform_points(points, matrix)
        forward = ref_tree.query(moved)[0]
        backward = cKDTree(moved).query(ref_points)[0]
        return float(np.mean(forward**2) + np.mean(backward**2))

    def surface_cost(matrix: Array) -> float:
        moved = trimesh.transform_points(points, matrix)
        inverse = trimesh.transform_points(ref_points, np.linalg.inv(matrix))
        forward = reference.nearest.on_surface(moved)[1]
        backward = prediction.nearest.on_surface(inverse)[1]
        return float(np.mean(forward**2) + np.mean(backward**2))

    progress(f"Совмещение: предварительный отбор из {len(candidates)} положений")
    ranked = sorted(enumerate(candidates), key=lambda item: (coarse_cost(item[1]), item[0]))
    best, best_cost = None, float("inf")
    records = []
    selected = ranked[: config.icp_candidates]
    for number, (index, initial) in enumerate(selected, start=1):
        progress(f"Совмещение: уточнение положения {number}/{len(selected)} (ICP)")
        refined, _, _ = trimesh.registration.icp(
            points,
            reference,
            initial=initial,
            scale=False,
            reflection=False,
            max_iterations=config.icp_iterations,
            threshold=1e-10,
        )
        # ICP оптимизирует одно направление: исходный кандидат иногда лучше симметрически.
        progress(f"Совмещение: проверка положения {number}/{len(selected)}")
        for stage, matrix in (("initial", initial), ("icp", refined)):
            validate_rigid(matrix)
            cost = surface_cost(matrix)
            records.append({"candidate": index, "stage": stage, "cd2_mm2": cost})
            if cost < best_cost:
                best, best_cost = matrix, cost
    assert best is not None
    return best, {
        "method": "kernel_axes_multistart_rigid_icp",
        "candidate_count": len(candidates),
        "refined_count": min(len(candidates), config.icp_candidates),
        "selection": "symmetric_registration_point_to_surface_cd2",
        "cost_mm2": best_cost,
        "candidates": records,
        "global_optimum_guaranteed": False,
    }
