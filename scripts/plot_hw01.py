"""Построить иллюстрации сохранённого baseline, не обращаясь к CAD или модели."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def save(figure: plt.Figure, path: Path) -> None:
    temporary = path.with_suffix(".pending.png")
    figure.savefig(temporary, dpi=160, bbox_inches="tight")
    temporary.replace(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/hw01/baseline-v1"))
    parser.add_argument("--output", type=Path, default=Path("docs/assignments/hw-01/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.bundle / "expected-metrics.json").read_text())
    cases = report["cases"]
    positions = np.arange(len(cases))
    figure, axis = plt.subplots(figsize=(7, 4))
    for offset, key, title, color in (
        (-0.18, "solid_iou", "IoU", "#2563eb"),
        (0.18, "f_score", "F-score", "#0f766e"),
    ):
        values = [case["primary"][key]["value"] for case in cases]
        bars = axis.bar(positions + offset, values, 0.34, label=title, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set(
        xticks=positions,
        xticklabels=[c["id"] for c in cases],
        ylim=(0, 1.12),
        ylabel="Значение метрики",
        title="Baseline · один запуск на деталь",
    )
    axis.legend(loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, args.output / "metrics.png")
    for case in cases:
        deviations = args.bundle / "metric-evidence" / case["id"] / "deviations.json"
        if not deviations.exists():
            continue
        data = json.loads(deviations.read_text())
        tau = case["aligned"]["surface"]["tau_mm"]
        figure = plt.figure(figsize=(12, 5.2))
        grid = figure.add_gridspec(1, 3, width_ratios=(1, 1, 0.045), wspace=0.28)
        reference = np.asarray(data["reference_points_mm"])
        # Тонкая ось вертикальна на рисунке; подписи сохраняют исходные имена осей.
        thin_axis = int(np.argmin(np.ptp(reference, axis=0)))
        order = [(thin_axis + 1) % 3, (thin_axis + 2) % 3, thin_axis]
        labels = [f"{'XYZ'[i]}, мм" for i in order]
        for index, (role, title) in enumerate((("reference", "Эталон"), ("prediction", "Результат"))):
            axis = figure.add_subplot(grid[0, index], projection="3d")
            points = np.asarray(data[f"{role}_points_mm"])[:, order]
            distance = np.asarray(data[f"{role}_distances_mm"])
            chart = axis.scatter(*points[::4].T, c=distance[::4] / tau, cmap="magma", vmin=0, vmax=2, s=2)
            axis.set_title(title)
            axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1))
            axis.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
            for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
                coordinate.set_major_locator(MaxNLocator(3))
            axis.tick_params(labelsize=9, pad=1)
        figure.colorbar(chart, cax=figure.add_subplot(grid[0, 2]), label="Расстояние / τ (цвет ограничен 2τ)")
        figure.suptitle(f"{case['id']} · поверхность после совмещения · τ={tau:.3f} мм")
        figure.subplots_adjust(top=0.83, bottom=0.13, left=0.03, right=0.91)
        save(figure, args.output / f"{case['id']}-deviations.png")


if __name__ == "__main__":
    main()
