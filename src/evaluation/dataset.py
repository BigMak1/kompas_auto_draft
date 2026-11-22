"""Проверка манифеста датасета: SHA256, однолистовость, геометрия, отсутствие утечек dev/val."""

from pathlib import Path

import numpy as np

from src.core.utils import read_json, require_orientation, require_single_sheet, sha256
from src.evaluation.metrics import load_mesh


def check_manifest(
    root: Path,
    manifest_relative: str,
    *,
    validation_size: int = 5,
    require_val: bool = True,
) -> dict:
    """Проверяет целостность файлов, координаты и отсутствие пересечения семейств между dev и val."""
    path = root / manifest_relative
    manifest = read_json(path)
    items = manifest["items"]
    names = [item["pair_id"] for item in items]
    if not names or len(names) != len(set(names)):
        raise ValueError("Манифест датасета должен содержать уникальные детали")
    if manifest["coordinates"]["units"] != "mm":
        raise ValueError("Ожидаются миллиметры")
    families: dict[str, set[str]] = {"dev": set(), "val": set()}
    hashes: dict[str, set[str]] = {"dev": set(), "val": set()}
    for item in items:
        require_single_sheet(item)
        require_orientation(item)
        split = item["split"]
        if split not in families or not item.get("family"):
            raise ValueError("Для каждой детали нужны dev/val и группа родства family")
        families[split].add(item["family"])
        for field, folder, extension in (("input_png", "input", "png"), ("target_mesh", "target", "stl")):
            entry = item[field]
            if entry["path"] != f"{folder}/{item['pair_id']}.{extension}":
                raise ValueError(f"Некорректный путь {field}: {entry['path']}")
            source = path.parent / entry["path"]
            if source.is_symlink() or not source.is_file() or sha256(source) != entry["sha256"]:
                raise ValueError(f"Файл отсутствует, подменён или изменён: {entry['path']}")
            hashes[split].add(entry["sha256"])
        mesh = load_mesh(path.parent / item["target_mesh"]["path"])
        if not mesh.is_volume or not np.isfinite(mesh.vertices).all() or abs(mesh.bounds[0, 2]) > 1e-5:
            raise ValueError(f"{item['pair_id']}: некорректное тело или Zmin != 0")
        if split == "val" and item.get("review", {}).get("status") != "accepted":
            raise ValueError(f"{item['pair_id']}: проверочная пара не принята")
        if item.get("preparation"):
            matrix = np.array(item["preparation"]["transform_to_target"], dtype=float)
            if (
                matrix.shape != (4, 4)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], (0, 0, 0, 1))
                or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3))
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)
            ):
                raise ValueError("Подготовка эталона допускает только поворот и перенос")
    if families["dev"] & families["val"] or hashes["dev"] & hashes["val"]:
        raise ValueError("Родственные детали или одинаковые файлы попали в dev и val")
    if require_val and (
        manifest.get("status") != "ready" or sum(item["split"] == "val" for item in items) != validation_size
    ):
        raise ValueError("Проверочная выборка ещё не подготовлена полностью")
    return manifest
