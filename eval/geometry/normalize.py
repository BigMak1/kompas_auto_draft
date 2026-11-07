"""Нормализация ответов ядра; МЦХ по сетке здесь не пересчитываются."""

from pathlib import Path

import numpy as np

from eval.geometry.io import atomic_json, read_json, sha256
from eval.geometry.models import Facts


def normalize_facts(data: dict, raw: dict) -> tuple[Facts, str]:
    axes = np.asarray(data.get("principal_axes"), dtype=float)
    if axes.shape == (3, 3) and np.allclose(axes.T @ axes, np.eye(3), atol=1e-6, rtol=0):
        return Facts.model_validate(data), "kernel_axes"
    # При кратных собственных значениях KsAPI возвращает нулевые оси и success=True.
    # Диагонализируем уже измеренный ядром центральный тензор, не интегрируя геометрию.
    x, y, z, xy, xz, yz = raw["central_moments"]
    tensor = np.array([[x, -xy, -xz], [-xy, y, -yz], [-xz, -yz, z]], dtype=float)
    if not np.isfinite(tensor).all():
        raise ValueError("Неконечный тензор инерции ядра")
    moments, basis = np.linalg.eigh(tensor)
    kernel_moments = np.sort(data["principal_moments"])
    if not np.allclose(moments, kernel_moments, atol=1e-8, rtol=1e-8):
        raise ValueError("Центральный тензор не согласован с главными моментами КОМПАС")
    normalized = {**data, "principal_axes": basis.tolist(), "principal_moments": kernel_moments.tolist()}
    return Facts.model_validate(normalized), "kernel_tensor_eigendecomposition"


def normalize_export(directory: Path) -> Facts:
    record = read_json(directory / "export-source.json")
    for name, expected in record["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Повреждён ответ вычислителя: {name}")
    facts, method = normalize_facts(
        read_json(directory / "kernel-facts.json"),
        read_json(directory / "raw-facts.json"),
    )
    atomic_json(directory / "eval-facts.json", facts.model_dump())
    atomic_json(
        directory / "export.json",
        {
            **record,
            "axes_method": method,
            "files": {**record["files"], "eval-facts.json": sha256(directory / "eval-facts.json")},
        },
    )
    return facts
