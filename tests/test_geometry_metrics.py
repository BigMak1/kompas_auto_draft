"""Проверка смысла метрик на независимых простых геометрических случаях."""

import unittest

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from evaluation.geometry import closest, measure, rigid_check, summarize_distances, topology


class GeometryMetricsTests(unittest.TestCase):
    def test_identical_surfaces(self) -> None:
        box = trimesh.creation.box(extents=[2, 3, 4])
        result, _ = measure(box, box, 300, 5)
        self.assertLess(result["symmetric_mean_mm"], 1e-12)
        self.assertEqual(result["thresholds"][0]["f1"], 1)

    def test_distance_to_triangle_interior(self) -> None:
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 0]], faces=[[0, 1, 2]])
        point, distances, _ = closest(mesh, np.array([[2.0, 3.0, 4.0]]))
        np.testing.assert_allclose(point, [[2, 3, 0]])
        np.testing.assert_allclose(distances, [4])

    def test_known_rigid_transform_preserves_shape(self) -> None:
        box = trimesh.creation.box(extents=[2, 3, 4])
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [13, 29, 41], degrees=True).as_matrix()
        transform[:3, 3] = [12, -8, 43]
        rigid_check(transform)
        restored = box.copy().apply_transform(transform).apply_transform(np.linalg.inv(transform))
        result, _ = measure(restored, box, 300, 6)
        self.assertLess(result["symmetric_mean_mm"], 1e-12)

    def test_scale_and_reflection_are_rejected(self) -> None:
        for change in (np.diag([2, 2, 2, 1]), np.diag([-1, 1, 1, 1])):
            with self.assertRaises(ValueError):
                rigid_check(change)

    def test_missing_material_has_directional_asymmetry(self) -> None:
        # Сфера меньшего радиуса: расстояния обеих поверхностей положительны,
        # поэтому тест не может пройти только за счёт удачного nearest-neighbor.
        ref = trimesh.creation.icosphere(subdivisions=2, radius=2)
        pred = ref.copy().apply_scale(0.5)
        result, _ = measure(pred, ref, 300, 7)
        self.assertGreater(result["symmetric_mean_mm"], 0.9)
        self.assertEqual(result["thresholds"][2]["f1"], 0)

    def test_f1_and_quantile_are_directional(self) -> None:
        result = summarize_distances(np.array([0.0, 0.0]), np.array([0.0, 2.0]), [0.1])
        self.assertAlmostEqual(result["thresholds"][0]["f1"], 2 / 3)
        self.assertEqual(result["symmetric_mean_mm"], 0.5)
        self.assertEqual(result["chamfer_squared_mean_mm2"], 1)
        self.assertAlmostEqual(result["symmetric_p95_mm"], 1.9)

    def test_through_hole_changes_topology(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=2)
        torus = trimesh.creation.torus(major_radius=2, minor_radius=0.5)
        self.assertEqual(topology(sphere)["boundary_genus_sum"], 0)
        self.assertEqual(topology(torus)["boundary_genus_sum"], 1)


if __name__ == "__main__":
    unittest.main()
