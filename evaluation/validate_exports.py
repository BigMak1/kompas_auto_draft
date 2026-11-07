"""Сравнение дискретизации при одной позе и воспроизводимой выборке."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from evaluation.geometry import measure, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    path = parser.parse_args().directory
    surface = json.loads((path / "surface-metrics.json").read_text())
    matrix = np.array(surface["alignment"]["matrix_prediction_to_reference"])
    native = path / surface["mesh_export_directory"]
    pred = trimesh.load_mesh(native / "prediction-0.01.stl").apply_transform(matrix)
    ref = trimesh.load_mesh(native / "reference-0.01.stl")
    result, _ = measure(pred, ref, 20000, 20261107)
    save_json(
        path / "tessellation-check.json",
        {
            "rigid_transform": matrix.tolist(),
            "chord_0.01_mm": result,
            "chord_0.003_mm": surface["surface_20000"],
        },
    )


if __name__ == "__main__":
    main()
