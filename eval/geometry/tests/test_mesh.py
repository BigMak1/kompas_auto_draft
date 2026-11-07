"""Численная склейка не чинит дефекты; пересечения устойчивы к положению детали."""

import sys

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from eval.geometry.diagnostics import mesh_diagnostics
from eval.geometry.mesh import normalize_vertices


def soup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=mesh.triangles.reshape(-1, 3).copy(),
        faces=np.arange(len(mesh.faces) * 3).reshape(-1, 3),
        process=False,
    )


def test_roundoff_duplicates_close_without_changing_faces() -> None:
    original = trimesh.creation.box(extents=[10, 20, 30])
    raw = soup(original)
    raw.vertices[0, 0] += 1e-14
    before = raw.vertices.copy()
    mesh, record = normalize_vertices(raw)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert len(mesh.faces) == len(original.faces)
    assert record["roundoff_merged_vertices"] == 1
    assert 0 < record["max_displacement_mm"] <= record["displacement_limit_mm"] <= 1e-9
    assert mesh.volume == pytest.approx(original.volume, rel=1e-14)
    np.testing.assert_array_equal(raw.vertices, before)
    # Представители взяты из входа; усреднения/сглаживания нет.
    assert all(np.any(np.all(before == vertex, axis=1)) for vertex in mesh.vertices)


def test_real_crack_hole_and_duplicates_are_preserved() -> None:
    raw = soup(trimesh.creation.box())
    raw.vertices[:3, 0] += 1e-7
    mesh, _ = normalize_vertices(raw)
    assert not mesh.is_watertight
    assert mesh_diagnostics(mesh)["dangling_edge_length_mm"] > 0
    opened = trimesh.creation.box()
    opened.update_faces(np.arange(len(opened.faces)) != 0)
    mesh, _ = normalize_vertices(soup(opened))
    assert not mesh.is_watertight
    box = trimesh.creation.box()
    duplicate = trimesh.Trimesh(vertices=box.vertices, faces=np.vstack([box.faces, box.faces[0]]))
    mesh, _ = normalize_vertices(duplicate)
    assert len(mesh.faces) == 13
    assert mesh_diagnostics(mesh)["nonmanifold_edges"] == 3


def test_normalization_rejects_collapsing_edge_and_long_chain() -> None:
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1e-15, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    with pytest.raises(ValueError, match="схлопнула"):
        normalize_vertices(mesh)
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1e-14, 0, 0], [2e-14, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 3], [1, 2, 3]],
        process=False,
    )
    with pytest.raises(ValueError, match="Цепочка"):
        normalize_vertices(mesh)


@pytest.mark.parametrize("seed", range(5))
def test_rotated_curved_mesh_and_crossing_bodies(seed: int) -> None:
    cylinder = trimesh.creation.cylinder(radius=20, height=33, sections=128)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.random(random_state=seed).as_matrix()
    matrix[:3, 3] = [13.1, -21.2, 36.3]
    cylinder.apply_transform(matrix)
    assert mesh_diagnostics(cylinder)["self_intersection_ratio"]["value"] == 0
    box = trimesh.creation.box()
    other = box.copy()
    other.apply_translation([0.25, 0.25, 0.25])
    intersecting = trimesh.util.concatenate([box, other])
    original = mesh_diagnostics(intersecting)["self_intersection_ratio"]["value"]
    intersecting.apply_transform(matrix)
    assert mesh_diagnostics(intersecting)["self_intersection_ratio"]["value"] == original > 0


def test_adjacent_triangles_can_really_overlap() -> None:
    # Общую вершину нельзя просто исключать из проверки: за ней есть пересечение.
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 1, -1], [1, 1, 1]],
        faces=[[0, 1, 2], [0, 3, 4]],
        process=False,
    )
    assert mesh_diagnostics(mesh)["self_intersection_ratio"]["value"] == 1
    # Совпланарные грани с общим ребром перекрываются по площади.
    coplanar = trimesh.Trimesh(
        vertices=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 1, 0]],
        faces=[[0, 1, 2], [0, 1, 3]],
        process=False,
    )
    assert mesh_diagnostics(coplanar)["self_intersection_ratio"]["value"] == 1


def test_unavailable_checker_is_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "igl.copyleft.cgal", None)
    diagnostic = mesh_diagnostics(trimesh.creation.box())["self_intersection_ratio"]
    assert diagnostic["status"] == "not_computed"
    assert diagnostic["value"] is None
