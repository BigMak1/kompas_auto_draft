"""Поверхностные метрики в миллиметрах; rigid alignment без отражения и масштаба."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def save_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def rigid_check(matrix: np.ndarray) -> None:
    if not (
        np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
        and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
        and np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-8)
    ):
        raise ValueError("Expected rigid transform without scaling or reflection")


def closest(mesh: trimesh.Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = [
        trimesh.proximity.closest_point(mesh, p) for p in np.array_split(points, max(1, len(points) // 1000))
    ]
    return tuple(np.concatenate([block[i] for block in blocks]) for i in range(3))


def align(pred: trimesh.Trimesh, ref: trimesh.Trimesh, seed: int = 20261106) -> dict:
    """Multi-start ICP: 24 principal-axis permutations + 144 axial starts, then triangle ICP.

    Первичный поиск и ICP по облакам дают кандидатов. Окончательный выбор — по сумме
    средних расстояний точка→треугольник в обоих направлениях на отдельных выборках.
    Это численная эвристика; доказательства глобального оптимума нет.
    """
    a, _ = trimesh.sample.sample_surface(pred, 6000, seed=seed)
    b, _ = trimesh.sample.sample_surface(ref, 100000, seed=seed + 1)
    ca, cb = pred.center_mass, ref.center_mass
    _, ua = np.linalg.eigh(pred.moment_inertia)
    _, ub = np.linalg.eigh(ref.moment_inertia)
    rotations = [np.eye(3)]
    for order in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            r = ub[:, order] @ np.diag(signs) @ ua.T
            if np.linalg.det(r) > 0:
                rotations.append(r)
    for flip in (np.eye(3), np.diag([1, -1, -1])):
        for angle in np.arange(0, 360, 5):
            r = ub @ Rotation.from_euler("z", angle, degrees=True).as_matrix() @ flip @ ua.T
            if np.linalg.det(r) < 0:
                r = (
                    ub
                    @ Rotation.from_euler("z", angle, degrees=True).as_matrix()
                    @ flip
                    @ np.diag([-1, 1, 1])
                    @ ua.T
                )
            rotations.append(r)
    tree = cKDTree(b)
    initial = []
    for r in rotations:
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = r, cb - r @ ca
        cost = float(np.mean(tree.query(trimesh.transform_points(a[::3], t))[0] ** 2))
        initial.append((cost, t))
    candidates = []
    for _, t in sorted(initial, key=lambda x: x[0])[:16]:
        result, _, cost = trimesh.registration.icp(
            a, b, initial=t, max_iterations=45, threshold=1e-8, scale=False, reflection=False
        )
        candidates.append((float(cost), result))
    # Учитываем точное совпадение исходных осей: ICP может сдвинуть правильные грани
    # ради компенсации ошибки формы. Это дополнительный проверяемый кандидат.
    centered = np.eye(4)
    centered[:3, 3] = ref.bounds.mean(axis=0) - pred.bounds.mean(axis=0)
    candidates.append((0.0, centered))
    validation_a, _ = trimesh.sample.sample_surface(pred, 6000, seed=seed + 2)
    validation_b, _ = trimesh.sample.sample_surface(ref, 6000, seed=seed + 3)
    finalists = []
    # Сохраняем согласованные оси, проверяя также нулевые сдвиги по отдельным осям.
    for mask in itertools.product([0, 1], repeat=3):
        fit = centered.copy()
        fit[:3, 3] *= mask
        moved = pred.copy().apply_transform(fit)
        d1 = closest(ref, trimesh.transform_points(validation_a, fit))[1]
        d2 = closest(moved, validation_b)[1]
        finalists.append(
            {
                "mean_distance_mm": float((d1.mean() + d2.mean()) / 2),
                "matrix": fit.tolist(),
                "triangle_refined": False,
                "source": "axis_translation",
            }
        )
    for _, t in sorted(candidates, key=lambda x: x[0])[:5]:
        for refine in (False, True):
            fit = t.copy()
            if refine:
                fit, _, _ = trimesh.registration.icp(
                    a, ref, initial=fit, max_iterations=18, threshold=1e-9, scale=False, reflection=False
                )
            rigid_check(fit)
            moved = pred.copy().apply_transform(fit)
            d1 = closest(ref, trimesh.transform_points(validation_a, fit))[1]
            d2 = closest(moved, validation_b)[1]
            score = float((d1.mean() + d2.mean()) / 2)
            finalists.append({"mean_distance_mm": score, "matrix": fit.tolist(), "triangle_refined": refine})
    best = min(finalists, key=lambda x: x["mean_distance_mm"])
    # L1-уточнение: стандартный ICP минимизирует квадрат ошибки и может сдвигать
    # совпадающие грани ради небольшой области с большой ошибкой формы.
    base = np.array(best["matrix"])
    center = ref.center_mass

    def transformed(delta: np.ndarray) -> np.ndarray:
        change = np.eye(4)
        change[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix()
        change[:3, 3] = center - change[:3, :3] @ center + delta[:3]
        return change @ base

    def objective(delta: np.ndarray) -> float:
        fit = transformed(delta)
        forward = closest(ref, trimesh.transform_points(validation_a[:1500], fit))[1]
        inverse = closest(pred, trimesh.transform_points(validation_b[:1500], np.linalg.inv(fit)))[1]
        return float((forward.mean() + inverse.mean()) / 2)

    optimized = minimize(
        objective, np.zeros(6), method="Powell", options={"maxiter": 5, "xtol": 1e-6, "ftol": 1e-6}
    )
    fit = transformed(optimized.x)
    rigid_check(fit)
    d1 = closest(ref, trimesh.transform_points(validation_a, fit))[1]
    d2 = closest(pred, trimesh.transform_points(validation_b, np.linalg.inv(fit)))[1]
    finalists.append(
        {
            "mean_distance_mm": float((d1.mean() + d2.mean()) / 2),
            "matrix": fit.tolist(),
            "triangle_refined": True,
            "source": "symmetric_L1_Powell",
            "optimizer_converged": bool(optimized.success),
        }
    )
    best = min(finalists, key=lambda x: x["mean_distance_mm"])
    # Изменение меньше 0.0001 мм существенно ниже точности используемой STL-сетки.
    # В таком случае предпочитаем сохранение осей микроповороту, найденному по шуму выборки.
    simple = [
        item
        for item in finalists
        if item.get("source") == "axis_translation"
        and item["mean_distance_mm"] <= best["mean_distance_mm"] + 0.0001
    ]
    if simple:
        best = min(simple, key=lambda x: x["mean_distance_mm"])
    return {
        "method": "multi_start_rigid_icp_triangle_validation",
        "seed": seed,
        "scale_allowed": False,
        "reflection_allowed": False,
        "global_optimum_proven": False,
        "initializations": len(rotations),
        "axis_preserving_tie_tolerance_mm": 0.0001,
        "matrix_prediction_to_reference": best["matrix"],
        "candidates": finalists,
    }


def summarize_distances(d1: np.ndarray, d2: np.ndarray, taus: list[float]) -> dict:
    def stats(d: np.ndarray) -> dict:
        return {
            "mean_mm": float(d.mean()),
            "rms_mm": float(np.sqrt(np.mean(d**2))),
            "p95_mm": float(np.quantile(d, 0.95)),
            "p99_mm": float(np.quantile(d, 0.99)),
            "sampled_max_mm": float(d.max()),
        }

    thresholds = []
    for tau in taus:
        precision, recall = float(np.mean(d1 <= tau)), float(np.mean(d2 <= tau))
        thresholds.append(
            {
                "tolerance_mm": tau,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    squared = float((np.mean(d1**2) + np.mean(d2**2)) / 2)
    return {
        "prediction_to_reference": stats(d1),
        "reference_to_prediction": stats(d2),
        "symmetric_mean_mm": float((d1.mean() + d2.mean()) / 2),
        "symmetric_rms_mm": float(np.sqrt(squared)),
        "chamfer_squared_mean_mm2": squared,
        "symmetric_p95_mm": float(max(np.quantile(d1, 0.95), np.quantile(d2, 0.95))),
        "sampled_hausdorff_lower_bound_mm": float(max(d1.max(), d2.max())),
        "thresholds": thresholds,
    }


def topology(mesh: trimesh.Trimesh) -> dict:
    components = mesh.split(only_watertight=False)
    valid = bool(mesh.is_watertight and mesh.is_winding_consistent)
    return {
        "triangles": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_shells": len(components),
        "euler_characteristic": int(mesh.euler_number),
        "boundary_genus_sum": int(sum(1 - c.euler_number / 2 for c in components)) if valid else None,
        "mesh_volume_mm3": float(mesh.volume),
        "mesh_area_mm2": float(mesh.area),
        "bbox_mm": mesh.bounds.tolist(),
        "extents_mm": mesh.extents.tolist(),
        "inertia_eigenvalues_unit_density_mm5": np.linalg.eigvalsh(mesh.moment_inertia).tolist(),
    }


def measure(pred: trimesh.Trimesh, ref: trimesh.Trimesh, count: int, seed: int) -> tuple[dict, dict]:
    p, pf = trimesh.sample.sample_surface(pred, count, seed=seed)
    r, rf = trimesh.sample.sample_surface(ref, count, seed=seed + 1)
    _, dp, nearest_ref = closest(ref, p)
    _, dr, nearest_pred = closest(pred, r)
    result = summarize_distances(dp, dr, [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    np_dot = np.einsum("ij,ij->i", pred.face_normals[pf], ref.face_normals[nearest_ref])
    nr_dot = np.einsum("ij,ij->i", ref.face_normals[rf], pred.face_normals[nearest_pred])
    result.update(
        {
            "samples_per_surface": count,
            "seed": seed,
            "normal_consistency_absolute": float((np.abs(np_dot).mean() + np.abs(nr_dot).mean()) / 2),
            "normal_consistency_oriented": float((np_dot.mean() + nr_dot.mean()) / 2),
        }
    )
    return result, {
        "prediction_points": p,
        "reference_points": r,
        "prediction_distances": dp,
        "reference_distances": dr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--native", default="native")
    parser.add_argument("--chord", default="0.003")
    args = parser.parse_args()
    folder = args.directory
    pred = trimesh.load_mesh(folder / args.native / f"prediction-{args.chord}.stl")
    ref = trimesh.load_mesh(folder / args.native / f"reference-{args.chord}.stl")
    alignment = align(pred, ref)
    save_json(folder / "alignment.json", alignment)
    pred.apply_transform(np.array(alignment["matrix_prediction_to_reference"]))
    result = {
        "units": "mm",
        "mesh_export_directory": args.native,
        "requested_chord_mm": float(args.chord),
        "alignment": alignment,
        "prediction": topology(pred),
        "reference": topology(ref),
        "acceptance_verdict": None,
        "acceptance_reason": "Engineering tolerances have not been agreed",
    }
    result["surface_20000"], _ = measure(pred, ref, 20000, 20261107)
    save_json(folder / "surface-metrics.json", result)
    result["surface_80000"], samples = measure(pred, ref, 80000, 20261108)
    save_json(folder / "surface-metrics.json", result)
    np.savez_compressed(folder / "surface-samples.npz", **samples)
    print(json.dumps(result["surface_80000"], indent=2), flush=True)


if __name__ == "__main__":
    main()
