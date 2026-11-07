"""Метрики поверхностей. Объёмный IoU здесь не аппроксимируется."""

from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from eval.geometry.models import Config
from eval.geometry.progress import ProgressCallback, no_progress

Array = NDArray[np.float64]


def coverage(forward: Array, backward: Array, tau: float) -> dict[str, float]:
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
    return {
        "precision": precision,
        "recall": recall,
        "f_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "siou": (precision + recall) / 2,
    }


def cloud_distances(prediction: Array, reference: Array) -> tuple[Array, Array]:
    for points in (prediction, reference):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("Ожидаются непустые конечные облака Nx3")
    return cKDTree(reference).query(prediction)[0], cKDTree(prediction).query(reference)[0]


def distance_metrics(forward: Array, backward: Array) -> dict[str, float]:
    coverage(forward, backward, 1)  # Проверка входа, включая пустые облака и NaN.
    return {
        "chamfer_l1_mm": float(np.mean(forward) + np.mean(backward)),
        "chamfer_l2_mm2": float(np.mean(forward**2) + np.mean(backward**2)),
        "hausdorff_mm": float(max(np.max(forward), np.max(backward))),
        "hausdorff95_mm": float(max(np.quantile(forward, 0.95), np.quantile(backward, 0.95))),
    }


def evaluate_surface(
    prediction: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    config: Config,
    length_mm: float,
    progress: ProgressCallback = no_progress,
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress("Выборка точек поверхности")
    tau = config.tau_mm if config.tau_mm is not None else config.tau_relative * length_mm
    # Семена регистрации и оценки различны; одинаковая сетка не получает общую выборку.
    pred_points, pred_faces = trimesh.sample.sample_surface(
        prediction, config.surface_samples, seed=config.seed + 100
    )
    ref_points, ref_faces = trimesh.sample.sample_surface(
        reference, config.surface_samples, seed=config.seed + 101
    )
    progress("Расстояния от предсказания до поверхности эталона")
    _, forward, nearest_ref = reference.nearest.on_surface(pred_points)
    progress("Расстояния от эталона до поверхности предсказания")
    _, backward, nearest_pred = prediction.nearest.on_surface(ref_points)
    progress("Расстояния между облаками точек (CD, HD, SIoU)")
    p2p_forward, p2p_backward = cloud_distances(pred_points, ref_points)
    progress("Normal Consistency: сравнение нормалей")
    dots_pred = np.einsum(
        "ij,ij->i", prediction.face_normals[pred_faces], reference.face_normals[nearest_ref]
    )
    dots_ref = np.einsum("ij,ij->i", reference.face_normals[ref_faces], prediction.face_normals[nearest_pred])
    progress("Chamfer, Hausdorff и HD95: точки и поверхности")
    p2p = distance_metrics(p2p_forward, p2p_backward)
    p2s = distance_metrics(forward, backward)
    progress("F-score и SIoU: покрытие поверхности в допуске")
    result = {
        **coverage(forward, backward, tau),
        "tau_mm": tau,
        "distance_mode": "point_to_triangle",
        "reference_length_mm": length_mm,
        "point_to_surface": p2s,
        "point_to_point": p2p,
        "chamfer_l2_reference_normalized": p2p["chamfer_l2_mm2"] / length_mm**2,
        "siou_cadbench_formula_p2p": coverage(p2p_forward, p2p_backward, 0.01 * length_mm)["siou"],
        "normal_consistency": float((np.mean(np.abs(dots_pred)) + np.mean(np.abs(dots_ref))) / 2),
        "normal_consistency_oriented": float((np.mean(dots_pred) + np.mean(dots_ref)) / 2),
    }
    deviations = {
        "prediction_points_mm": pred_points.tolist(),
        "prediction_distances_mm": forward.tolist(),
        "reference_points_mm": ref_points.tolist(),
        "reference_distances_mm": backward.tolist(),
    }
    return result, deviations


def sample_mesh_edges(mesh: trimesh.Trimesh, config: Config, seed: int) -> Array | None:
    """Явно помечаем proxy: резкие mesh-рёбра не равны всем B-rep-кромкам."""
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    edges = mesh.edges_unique[counts == 1]
    sharp = mesh.face_adjacency_edges[mesh.face_adjacency_angles >= np.deg2rad(config.edge_angle_degrees)]
    edges = np.concatenate((edges, sharp))
    if not len(edges):
        return None
    segments = mesh.vertices[edges]
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    if lengths.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    chosen = segments[rng.choice(len(segments), size=config.edge_samples, p=lengths / lengths.sum())]
    return chosen[:, 0] + rng.random((config.edge_samples, 1)) * (chosen[:, 1] - chosen[:, 0])
