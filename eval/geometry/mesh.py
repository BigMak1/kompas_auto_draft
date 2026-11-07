"""Сборка индексов STL с учётом округления float64, без ремонта поверхности."""

from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def normalize_vertices(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Вход уже в мм. Представитель группы — исходная вершина, не среднее.

    Предел — 64 epsilon float64 от масштаба координат, максимум 1e-9 мм.
    Это политика численного чтения, а не допуск CAD или параметр тесселяции.
    Цепочки близких точек не должны сдвигать вершины дальше этого предела.
    Если склейка схлопнула бы ребро грани, вход отвергается, а не чинится.
    """
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    faces = inverse[mesh.faces]
    limit = min(64 * np.finfo(np.float64).eps * max(1.0, float(np.abs(vertices).max())), 1e-9)
    pairs = cKDTree(vertices).query_pairs(limit, output_type="ndarray")
    representatives = np.arange(len(vertices))
    if len(pairs):
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(vertices), len(vertices))
        )
        count, labels = connected_components(graph, directed=False)
        first = np.full(count, len(vertices), dtype=np.int64)
        np.minimum.at(first, labels, np.arange(len(vertices)))
        representatives = first[labels]
    displacement = np.linalg.norm(vertices - vertices[representatives], axis=1)
    if displacement.max() > limit:
        raise ValueError("Цепочка близких вершин превышает предел численной нормализации")
    mapped = representatives[faces]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        if np.any((faces[:, i] != faces[:, j]) & (mapped[:, i] == mapped[:, j])):
            raise ValueError("Численная нормализация схлопнула бы ребро грани")
    retained, remap = np.unique(representatives, return_inverse=True)
    normalized = trimesh.Trimesh(vertices=vertices[retained], faces=remap[faces], process=False)
    return normalized, {
        "policy": "float64-roundoff-v1",
        "input_vertices": len(mesh.vertices),
        "exact_unique_vertices": len(vertices),
        "normalized_vertices": len(retained),
        "roundoff_merged_vertices": len(vertices) - len(retained),
        "displacement_limit_mm": limit,
        "max_displacement_mm": float(displacement.max()),
    }
