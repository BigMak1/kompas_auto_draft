"""Локальная диагностика KF0011 по геометрии отверстия, отдельно от общей метрики."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from evaluation.geometry import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    path = parser.parse_args().directory
    out = {}
    for label in ("prediction", "reference"):
        mesh = trimesh.load_mesh(path / "native-v2" / f"{label}-0.003.stl")
        center_z = mesh.bounds.mean(axis=0)[2]
        vertices = mesh.vertices
        mask = (
            (np.abs(vertices[:, 2] - center_z) < 15.6)
            & (np.linalg.norm(vertices[:, :2], axis=1) < 18)
            & (vertices[:, 1] < -1)
        )
        xy = vertices[mask, :2]
        if len(xy) < 3:
            raise ValueError("Not enough bore vertices")
        rhs = np.sum(xy * xy, axis=1)
        fit = np.linalg.lstsq(np.column_stack((2 * xy, np.ones(len(xy)))), rhs, rcond=None)[0]
        radius = np.sqrt(fit[2] + np.sum(fit[:2] ** 2))
        residual = np.linalg.norm(xy - fit[:2], axis=1) - radius
        out[label] = {
            "method": "least_squares_circle_STL_vertices_bore_negative_Y",
            "vertices": len(xy),
            "center_xy_mm": fit[:2].tolist(),
            "diameter_mm": float(2 * radius),
            "max_radial_fit_residual_mm": float(np.max(np.abs(residual))),
            "tessellation_chord_mm": 0.003,
        }
    out["reference_native_cylinder"] = json.loads((path / "local-native-features.json").read_text())[
        "reference"
    ]["central_cylinders"][0]
    out["drawing_manual_read"] = {
        "diameter_nominal_mm": 32,
        "diameter_upper_deviation_mm": 0.039,
        "diameter_lower_deviation_mm": 0,
        "keyway_width_mm": 10,
        "keyway_overall_height_mm": 35.3,
        "source": "../../builder/input/drawing.png",
    }
    save_json(path / "bore-measurements.json", out)


if __name__ == "__main__":
    main()
