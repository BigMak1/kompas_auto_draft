"""Визуализация результатов исследования: метрики, детали, ресурсы, ошибки и сечения STL."""

from pathlib import Path

OUTCOME_STYLES = (
    ("valid_stl", "Успех (валидный STL)", "#2e8b57"),
    ("invalid_stl", "Незамкнутый STL", "#e69f00"),
    ("cad_error", "Ошибка CadQuery", "#d55e00"),
    ("token_limit", "Лимит токенов", "#cc79a7"),
    ("schema_or_json", "Ошибка JSON / схемы", "#0072b2"),
    ("infrastructure", "Инфраструктура / пропуск", "#7f7f7f"),
)


def _short_label(label: str) -> str:
    if " · " in label:
        return label.replace(" · ", "\n", 1)
    return label.replace(" ", "\n", 1)


def summary_plots(rows: list[dict], summary: dict, study: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = list(summary["variants"])
    labels = [study["config"]["variants"][v]["label"] for v in variants]
    short_labels = [_short_label(label) for label in labels]

    # 1. Распределение и среднее IoU / F-score.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax, metric, title in zip(
        axes, ("iou", "f_score"), ("IoU (объём)", "F-score@τ (поверхность)"), strict=True
    ):
        for idx, variant in enumerate(variants):
            samples = [r[metric] for r in rows if r["variant"] == variant and r[metric] is not None]
            if samples:
                jitter = np.linspace(-0.08, 0.08, len(samples)) if len(samples) > 1 else [0.0]
                ax.scatter([idx + j for j in jitter], samples, alpha=0.45, s=24, color="#4c72b0")
            mean_val = summary["variants"][variant][metric]
            if mean_val is not None:
                ax.scatter(idx, mean_val, marker="D", color="#111111", s=70, zorder=3)
                ax.annotate(
                    f"{mean_val:.3f}",
                    (idx, mean_val),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8.5,
                    fontweight="bold",
                )
        ax.set(
            xticks=range(len(variants)),
            xticklabels=short_labels,
            ylim=(-0.05, 1.08),
            ylabel="IoU" if metric == "iou" else "F-score",
            title=title,
        )
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Точки — отдельные попытки; чёрный ромб — среднее по деталям полной оценки")
    fig.savefig(output / "metrics.png", dpi=180)
    plt.close(fig)

    # 2. Разбивка по деталям (per-part grouped bar chart).
    pairs = study["pairs"]
    if pairs and variants:
        width = 0.8 / max(1, len(variants))
        x = np.arange(len(pairs))
        fig, axes = plt.subplots(1, 2, figsize=(max(11, 2.2 * len(pairs)), 4.6), constrained_layout=True)
        for ax, metric, title in zip(
            axes, ("iou", "f_score"), ("IoU по деталям", "F-score по деталям"), strict=True
        ):
            for v_idx, (variant, label) in enumerate(zip(variants, labels, strict=True)):
                heights = []
                for pair in pairs:
                    vals = [
                        r[metric]
                        for r in rows
                        if r["variant"] == variant and r["pair_id"] == pair and r[metric] is not None
                    ]
                    heights.append(sum(vals) / len(vals) if vals else 0.0)
                offset = (v_idx - (len(variants) - 1) / 2) * width
                ax.bar(x + offset, heights, width=width * 0.92, label=label)
            ax.set(xticks=x, xticklabels=pairs, ylim=(0, 1.08), ylabel=title)
            ax.grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8, loc="upper right")
        fig.suptitle("Сравнение маршрутов по каждой детали (среднее по повторам)")
        fig.savefig(output / "per_part_metrics.png", dpi=180)
        plt.close(fig)

    # 3. Затраты времени и выходных токенов.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, key, label in zip(
        axes,
        ("seconds_mean", "output_tokens_mean"),
        ("Среднее время, с", "Средние выходные токены"),
        strict=True,
    ):
        for idx, variant in enumerate(variants):
            value = summary["variants"][variant][key]
            if value is not None:
                bar = ax.bar(idx, value, color="#4c72b0")
                ax.bar_label(bar, fmt="%.1f" if key == "seconds_mean" else "%.0f", padding=3, fontsize=8.5)
        ax.set(xticks=range(len(variants)), xticklabels=short_labels, ylabel=label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Затраты на один маршрут (общий паспорт включён в использующие его маршруты)")
    fig.savefig(output / "resources.png", dpi=180)
    plt.close(fig)

    # 4. Разбивка исходов и ошибок (stacked bar chart).
    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    bottom = np.zeros(len(variants))
    for cat_key, cat_label, color in OUTCOME_STYLES:
        counts = np.array(
            [summary["variants"][v]["outcome_counts"].get(cat_key, 0) for v in variants],
            dtype=float,
        )
        if counts.sum() > 0:
            ax.bar(range(len(variants)), counts, bottom=bottom, label=cat_label, color=color)
            bottom += counts
    ax.set(xticks=range(len(variants)), xticklabels=short_labels, ylabel="Число попыток")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    fig.suptitle("Структура исходов и причин сбоев по маршрутам")
    fig.savefig(output / "errors_breakdown.png", dpi=180)
    plt.close(fig)


def geometry_plots(root: Path, study_dir: Path, study: dict, rows: list[dict], output: Path) -> list[str]:
    """Сечения первого повтора каждой детали: 25%, 50%, 75% габарита эталона."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh
    from matplotlib.collections import LineCollection

    from src.evaluation.metrics import load_mesh

    files: list[str] = []
    variants_cfg = study["config"]["variants"]
    for pair in study["pairs"]:
        selected = [r for r in rows if r["pair_id"] == pair and r["repeat"] == 1]
        if not any(r["valid"] for r in selected):
            continue
        item = next(item for item in study["manifest"]["items"] if item["pair_id"] == pair)
        reference = load_mesh((root / study["config"]["manifest"]).parent / item["target_mesh"]["path"])
        predictions = {
            row["variant"]: load_mesh(study_dir / row["attempt"] / "model.stl") if row["valid"] else None
            for row in selected
        }
        meshes = [reference, *(mesh for mesh in predictions.values() if mesh is not None)]
        bounds = np.array(
            [np.min([m.bounds[0] for m in meshes], axis=0), np.max([m.bounds[1] for m in meshes], axis=0)]
        )
        fig, axes = plt.subplots(
            len(selected), 3, figsize=(10, 3 * len(selected)), squeeze=False, constrained_layout=True
        )
        for row_index, row in enumerate(selected):
            prediction = predictions[row["variant"]]
            label = variants_cfg[row["variant"]]["label"]
            for axis, normal_axis in zip(axes[row_index], (2, 1, 0), strict=True):
                projected = [i for i in range(3) if i != normal_axis]
                for mesh, color in ((reference, "#777777"), (prediction, "#1877b8")):
                    if mesh is None:
                        continue
                    for fraction in (0.25, 0.5, 0.75):
                        origin = reference.bounds.mean(axis=0)
                        origin[normal_axis] = (
                            reference.bounds[0, normal_axis] + fraction * reference.extents[normal_axis]
                        )
                        segments = trimesh.intersections.mesh_plane(mesh, np.eye(3)[normal_axis], origin)
                        if len(segments):
                            axis.add_collection(
                                LineCollection(
                                    segments[:, :, projected], colors=color, alpha=0.65, linewidths=0.8
                                )
                            )
                margin = max(reference.extents) * 0.05
                axis.set(
                    xlim=(bounds[0, projected[0]] - margin, bounds[1, projected[0]] + margin),
                    ylim=(bounds[0, projected[1]] - margin, bounds[1, projected[1]] + margin),
                    xlabel="XYZ"[projected[0]] + ", мм",
                    ylabel="XYZ"[projected[1]] + ", мм",
                    title=label,
                )
                axis.set_aspect("equal")
                if prediction is None:
                    axis.text(
                        0.5,
                        0.5,
                        "Нет измеренного результата",
                        ha="center",
                        transform=axis.transAxes,
                        fontsize=8,
                    )
        fig.suptitle(
            f"{pair}, повтор 1 · серый — эталон, синий — результат\n"
            "Сечения на 25%, 50%, 75% габарита эталона; совмещение отсутствует"
        )
        filename = f"geometry-{pair}.png"
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        files.append(filename)
    return files
