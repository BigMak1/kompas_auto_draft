"""Конфигурация исследования и проверка манифеста датасета dev/val."""

from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import yaml
from pydantic import Field, model_validator

from src.config import ROOT, AgentSettings, CadSettings, EvaluationSettings, Settings
from src.utils import read_json, require_orientation, require_single_sheet, sha256

CONFIG_DIRECTORY = ROOT / "configs/experiments"
DEFAULT_CONFIG = CONFIG_DIRECTORY / "01-drawing-passport.yaml"

Variant = Literal["direct", "passport_only", "passport_png", "notes_png"]
VARIANTS = ("direct", "passport_only", "passport_png", "notes_png")
LABELS = {
    "direct": "Прямой baseline",
    "passport_only": "Только паспорт",
    "passport_png": "Паспорт + PNG",
    "notes_png": "Заметки + PNG",
}
DESCRIPTIONS = {
    "direct": "PNG → рецепт → STL",
    "passport_only": "PNG → паспорт → рецепт → STL; построитель не видит PNG",
    "passport_png": "PNG → паспорт; паспорт + PNG → рецепт → STL",
    "notes_png": "PNG → заметки; заметки + PNG → рецепт → STL",
}
SafeName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]


class ReadingSettings(Settings):
    max_output_tokens: int = Field(gt=0)
    # Поля сохраняют совместимость с замороженными сериями до Prompt Registry.
    prompt_version: str = ""
    prompt: str = ""
    passport_format: str = ""
    notes_format: str = ""


class ExperimentAgentSettings(AgentSettings):
    prompt_version: str = ""
    prompt: str = ""


class PromptVersions(Settings):
    builder: str = Field(pattern=r"^prompts:/cad-builder/[1-9][0-9]*$")
    passport: str = Field(pattern=r"^prompts:/drawing-passport/[1-9][0-9]*$")
    notes: str = Field(pattern=r"^prompts:/drawing-notes/[1-9][0-9]*$")


class ExperimentConfig(Settings):
    number: int | None = Field(default=None, ge=1)
    title: str = "Сравнение маршрутов генерации CAD"
    description: str = ""
    name: SafeName
    manifest: str
    dataset_version: SafeName | None = None
    agent_version: SafeName | None = None
    prompts: PromptVersions | None = None
    prompt_snapshot: dict[str, dict] = Field(default_factory=dict)
    variants: list[Variant]
    repeats: int = Field(default=3, ge=1, le=100)
    validation_size: int = Field(default=5, ge=1)
    order_seed: int = Field(default=20261115, ge=0)
    progress_interval_seconds: float = Field(default=30, gt=0)
    agent: ExperimentAgentSettings
    reading: ReadingSettings
    cad: CadSettings
    evaluation: EvaluationSettings

    @model_validator(mode="after")
    def budget(self) -> Self:
        if self.prompts is not None and (not self.agent_version or not self.dataset_version):
            raise ValueError("Для Prompt Registry нужны явные agent_version и dataset_version")
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("Маршруты должны быть непустыми и не повторяться")
        if self.reading.max_output_tokens >= self.agent.max_output_tokens:
            raise ValueError("Чтение должно оставлять выходной бюджет для рецепта")
        if Path(self.manifest).is_absolute() or ".." in Path(self.manifest).parts:
            raise ValueError("Путь манифеста датасета должен быть относительным внутри репозитория")
        return self


def load_experiment(path: Path = DEFAULT_CONFIG) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def catalogue() -> list[tuple[Path, ExperimentConfig]]:
    entries = [(path, load_experiment(path)) for path in sorted(CONFIG_DIRECTORY.glob("*.yaml"))]
    numbers = [config.number for _, config in entries]
    if None in numbers or len(numbers) != len(set(numbers)):
        raise ValueError("В каталоге каждому исследованию нужен свой уникальный number")
    return sorted(entries, key=lambda entry: entry[1].number)


def resolve_experiment(selector: str | None, path: Path | None) -> tuple[Path, ExperimentConfig]:
    if path is not None:
        if selector is not None:
            raise ValueError("Выберите номер эксперимента или --config, не оба сразу")
        return path, load_experiment(path)
    if selector is None:
        return DEFAULT_CONFIG, load_experiment()
    matches = [
        (file, config)
        for file, config in catalogue()
        if selector in {str(config.number), file.stem, config.name}
    ]
    if len(matches) != 1:
        raise ValueError(f"Эксперимент {selector} не найден. Каталог: uv run poe experiment list")
    return matches[0]


def default_study(config: ExperimentConfig, split: str, pairs: list[str] | None, repeats: int) -> str:
    subset = "-".join(sorted({pair.lower() for pair in pairs})) if pairs else "all"
    return f"{config.name}-{split}-{subset}-r{repeats}"


def check_manifest(root: Path, config: ExperimentConfig, *, require_val: bool = True) -> dict:
    """Ни исходный корпус, ни КОМПАС для проверки не требуются."""
    from src.evaluate import load_mesh

    path = root / config.manifest
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
            raise ValueError("Для каждой детали нужны dev/val и группа родства")
        families[split].add(item["family"])
        for field, folder, extension in (("input_png", "input", "png"), ("target_mesh", "target", "stl")):
            entry = item[field]
            if entry["path"] != f"{folder}/{item['pair_id']}.{extension}":
                raise ValueError(f"Некорректный путь {field}")
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
        manifest.get("status") != "ready"
        or sum(item["split"] == "val" for item in items) != config.validation_size
    ):
        raise ValueError("Проверочная выборка ещё не подготовлена полностью")
    return manifest
