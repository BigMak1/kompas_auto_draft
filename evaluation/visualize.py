"""Сечения, поверхности и карта отклонений из сохранённых результатов оценки."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory
    result = json.loads((path / "surface-metrics.json").read_text())
    matrix = np.array(result["alignment"]["matrix_prediction_to_reference"])
    meshes = [
        trimesh.load_mesh(path / result["mesh_export_directory"] / f"{label}-0.003.stl")
        for label in ("reference", "prediction")
    ]
    meshes[1].apply_transform(matrix)
    samples = np.load(path / "surface-samples.npz")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(14, 11), facecolor="white", layout="constrained")
    fig.suptitle("KF0011 · сравнение итоговой модели с эталоном", fontsize=20, weight="bold")
    colors = ["#4779af", "#e88836"]
    for i, (mesh, title) in enumerate(zip(meshes, ["Эталон", "Модель агента · попытка №7"], strict=True)):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        light = np.array([-0.3, -0.5, 0.8])
        light /= np.linalg.norm(light)
        intensity = 0.4 + 0.6 * np.clip(mesh.face_normals @ light, 0, 1)
        shading = intensity[:, None] * np.array(to_rgb(colors[i]))[None, :]
        collection = Poly3DCollection(mesh.triangles, facecolors=shading, edgecolor="none", antialiased=False)
        ax.add_collection3d(collection)
        ax.set(xlim=(-38, 38), ylim=(-38, 38), zlim=(-2, 35), title=title)
        ax.set_box_aspect((76, 76, 37))
        ax.view_init(elev=52, azim=-65)
        ax.set_axis_off()
    ax = fig.add_subplot(2, 2, 3)
    for mesh, color, label in zip(meshes, colors, ["Эталон", "Модель агента"], strict=True):
        section = mesh.section(plane_origin=[0, 0, 16.5], plane_normal=[0, 0, 1])
        for index, entity in enumerate(section.entities):
            line = entity.discrete(section.vertices)
            ax.plot(line[:, 0], line[:, 1], color=color, lw=1.25, label=label if index == 0 else None)
    ax.set(title="Наложение сечений в середине детали", xlabel="X, мм", ylabel="Y, мм", aspect="equal")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    ax = fig.add_subplot(2, 2, 4)
    points, distances = samples["reference_points"], samples["reference_distances"]
    # Вид на середину венца исключает торцы, которые закрывают радиальные ошибки.
    selected = np.abs(points[:, 2] - 16.5) < 7
    scatter = ax.scatter(
        points[selected, 0],
        points[selected, 1],
        c=distances[selected],
        s=3,
        cmap="magma",
        vmin=0,
        vmax=max(0.1, float(np.quantile(distances, 0.999))),
    )
    ax.set(title="Отклонение от эталона на среднем поясе", xlabel="X, мм", ylabel="Y, мм", aspect="equal")
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="Расстояние до поверхности модели, мм")
    fig.savefig(path / "comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for key, color, label in [
        ("prediction_distances", colors[1], "Модель → эталон"),
        ("reference_distances", colors[0], "Эталон → модель"),
    ]:
        d = np.sort(samples[key])
        ax.plot(d, np.arange(1, len(d) + 1) / len(d) * 100, color=color, label=label)
    ax.set(
        xlabel="Допуск, мм",
        ylabel="Доля поверхности в допуске, %",
        ylim=(0, 101),
        title="KF0011 · накопленная доля поверхности по расстоянию",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path / "coverage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
