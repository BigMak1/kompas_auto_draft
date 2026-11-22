"""Декларативные настройки экспериментов, промпты как код и каталог YAML."""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.utils import ROOT, sha256_bytes

CONFIG_DIRECTORY = ROOT / "configs/experiments"
DEFAULT_CONFIG = CONFIG_DIRECTORY / "01-drawing-passport"
STUDY_MANIFEST_FILE = "study.yaml"
VARIANT_CONFIG_FILES = (
    "llm.yaml",
    "cad.yaml",
    "evaluation.yaml",
)
SHARED_SECTIONS = ("agent", "reading", "cad", "evaluation")

SafeName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
VariantRole = Literal["baseline", "improvement", "ablation"]


class StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _check_relative_repo_path(raw: str, label: str) -> None:
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} должен быть относительным путём внутри репозитория: {raw!r}")


class ExecutionSettings(StrictSettings):
    repeats: int = Field(default=3, ge=1, le=100)
    validation_size: int = Field(default=5, ge=1)
    order_seed: int = Field(default=20261115, ge=0)
    concurrency: int = Field(default=4, ge=1, le=32)
    progress_interval_seconds: float = Field(default=30.0, gt=0)


class AgentSettings(StrictSettings):
    model: str = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high"]
    timeout_seconds: float = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    image_detail: Literal["low", "high", "auto"]
    image_max_side: int | None = Field(default=None, ge=1)


class ReadingSettings(StrictSettings):
    max_output_tokens: int = Field(gt=0)


class CadSettings(StrictSettings):
    stl_tolerance_mm: float = Field(gt=0)
    stl_angle_rad: float = Field(gt=0)


class EvaluationSettings(StrictSettings):
    surface_samples: int = Field(ge=32)
    seed: int = Field(ge=0)
    tau_relative: float = Field(gt=0, le=1)
    intersection_roundoff_relative: float = Field(gt=0, le=1e-4)


class VariantConfig(StrictSettings):
    pipeline: SafeName
    role: VariantRole
    ablation_of: SafeName | None = None
    label: str = Field(min_length=1)
    description: str = ""
    hypothesis: str = ""
    prompts: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prompts(self) -> Self:
        for role, raw in self.prompts.items():
            if not role.strip():
                raise ValueError("Роль промпта не должна быть пустой")
            _check_relative_repo_path(raw, f"Путь промпта '{role}'")
        return self


class ComparisonConfig(StrictSettings):
    candidate: SafeName
    control: SafeName
    question: str = ""


class ExperimentConfig(StrictSettings):
    number: int | None = Field(default=None, ge=1)
    name: SafeName
    title: str = Field(min_length=1)
    description: str = ""
    hypothesis: str = ""
    dataset_version: SafeName
    manifest: str
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    agent: AgentSettings
    reading: ReadingSettings
    cad: CadSettings
    evaluation: EvaluationSettings
    variants: dict[SafeName, VariantConfig]
    comparisons: list[ComparisonConfig] = Field(default_factory=list)
    prompt_snapshot: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        _check_relative_repo_path(self.manifest, "Путь манифеста")
        if not self.variants:
            raise ValueError("В конфиге должен быть указан хотя бы один вариант")
        if self.reading.max_output_tokens >= self.agent.max_output_tokens:
            raise ValueError("Лимит чтения должен оставлять выходной бюджет для построителя рецепта")
        for key, variant in self.variants.items():
            if variant.ablation_of is not None:
                if variant.ablation_of == key:
                    raise ValueError(f"Вариант {key} не может ссылаться на себя в ablation_of")
                if variant.ablation_of not in self.variants:
                    raise ValueError(
                        f"Вариант {key} ссылается на неизвестный ablation_of={variant.ablation_of}"
                    )
        for pair in self.comparisons:
            if pair.candidate not in self.variants or pair.control not in self.variants:
                raise ValueError(
                    f"Сравнение {pair.candidate} vs {pair.control} ссылается на несуществующий вариант"
                )
            if pair.candidate == pair.control:
                raise ValueError(f"Нельзя сравнивать вариант {pair.candidate} с самим собой")
        return self


def _load_yaml_dict(file_path: Path, *, expected_root: Path | None = None) -> dict:
    if file_path.is_symlink():
        raise ValueError(f"Файл конфигурации не должен быть символической ссылкой: {file_path}")
    if expected_root is not None and not file_path.resolve().is_relative_to(expected_root.resolve()):
        raise ValueError(f"Файл конфигурации вне каталога эксперимента: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Отсутствует файл конфигурации: {file_path}")
    loaded = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Конфиг должен содержать YAML-словарь: {file_path}")
    return loaded


def _load_variant_folder(variant_dir: Path, study_root: Path) -> tuple[dict, dict]:
    """Читает конфиги одного эксперимента из подпапки (<variant>/*.yaml)."""
    if variant_dir.is_symlink() or not variant_dir.resolve().is_relative_to(study_root.resolve()):
        raise ValueError(f"Папка эксперимента не должна быть символической ссылкой: {variant_dir}")

    pipeline_data = _load_yaml_dict(variant_dir / "pipeline.yaml", expected_root=study_root)
    shared: dict = {}
    for filename in VARIANT_CONFIG_FILES:
        part_path = variant_dir / filename
        part = _load_yaml_dict(part_path, expected_root=study_root)
        overlap = set(shared) & set(part)
        if overlap:
            keys = ", ".join(sorted(overlap))
            raise ValueError(f"Дублирующиеся ключи ({keys}) в {part_path}")
        shared.update(part)

    missing = [section for section in SHARED_SECTIONS if section not in shared]
    if missing:
        raise ValueError(f"В папке эксперимента {variant_dir.name} отсутствуют секции: {', '.join(missing)}")
    return pipeline_data, shared


def _ensure_shared_sections_match(
    *,
    reference_name: str,
    reference: dict,
    candidate_name: str,
    candidate: dict,
) -> None:
    """Гарантирует единые условия LLM, CAD и оценки во всех папках экспериментов."""
    for section in SHARED_SECTIONS:
        ref_value = reference[section]
        cur_value = candidate[section]
        if cur_value == ref_value:
            continue
        if isinstance(ref_value, dict) and isinstance(cur_value, dict):
            diff_keys = sorted(
                key for key in set(ref_value) | set(cur_value) if ref_value.get(key) != cur_value.get(key)
            )
            detail = f" ({', '.join(diff_keys)})" if diff_keys else ""
        else:
            detail = ""
        raise ValueError(
            f"Нарушено честное сравнение: секция '{section}'{detail} в {candidate_name} "
            f"отличается от {reference_name}"
        )


def load_experiment(path: Path = DEFAULT_CONFIG) -> ExperimentConfig:
    if path.is_file() and path.name != STUDY_MANIFEST_FILE:
        return ExperimentConfig.model_validate(_load_yaml_dict(path))

    directory = path.parent if path.is_file() and path.name == STUDY_MANIFEST_FILE else path
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Отсутствует папка конфигурации исследования: {directory}")

    study_file = directory / STUDY_MANIFEST_FILE
    study = _load_yaml_dict(study_file, expected_root=directory)
    disallowed = sorted((set(SHARED_SECTIONS) | {"variants"}) & set(study))
    if disallowed:
        raise ValueError(
            f"В {STUDY_MANIFEST_FILE} не должно быть секций экспериментов "
            f"({', '.join(disallowed)}): {study_file}"
        )

    variant_dirs = [
        item
        for item in sorted(directory.iterdir())
        if not item.name.startswith(".") and (item.is_symlink() or item.is_dir())
    ]
    if not variant_dirs:
        raise ValueError(f"В папке исследования нет подпапок экспериментов: {directory}")

    variants: dict[str, dict] = {}
    shared_reference: dict | None = None
    reference_name = ""
    for variant_dir in variant_dirs:
        pipeline_data, variant_shared = _load_variant_folder(variant_dir, directory)
        variants[variant_dir.name] = pipeline_data
        if shared_reference is None:
            shared_reference = variant_shared
            reference_name = variant_dir.name
            continue
        _ensure_shared_sections_match(
            reference_name=reference_name,
            reference=shared_reference,
            candidate_name=variant_dir.name,
            candidate=variant_shared,
        )

    assert shared_reference is not None
    return ExperimentConfig.model_validate({**study, **shared_reference, "variants": variants})


def resolve_prompts(root: Path, config: ExperimentConfig) -> ExperimentConfig:
    """Считывает все промпты из репозитория, проверяет целостность и снимает snapshot."""
    root_resolved = root.resolve()
    snapshot: dict[str, dict[str, str]] = {}
    unique_paths = sorted({raw for variant in config.variants.values() for raw in variant.prompts.values()})
    for raw in unique_paths:
        path = root / raw
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_relative_to(root_resolved):
            raise ValueError(f"Недопустимый файл промпта (symlink или вне репозитория): {raw}")
        if not path.is_file():
            raise ValueError(f"Отсутствует файл промпта: {raw}")
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
        if not text.strip():
            raise ValueError(f"Файл промпта пустой: {raw}")
        snapshot[raw] = {
            "path": raw,
            "sha256": sha256_bytes(raw_bytes),
            "text": text,
        }
    return config.model_copy(update={"prompt_snapshot": snapshot})


def catalogue(directory: Path = CONFIG_DIRECTORY) -> list[tuple[Path, ExperimentConfig]]:
    entries: list[tuple[Path, ExperimentConfig]] = []
    for item in sorted(directory.iterdir()):
        if item.is_dir() and (item / STUDY_MANIFEST_FILE).is_file():
            entries.append((item, load_experiment(item)))
        elif item.is_file() and item.suffix == ".yaml":
            entries.append((item, load_experiment(item)))
    numbers = [config.number for _, config in entries]
    if None in numbers or len(numbers) != len(set(numbers)):
        raise ValueError("В каталоге каждому исследованию нужен свой уникальный number")
    return sorted(entries, key=lambda entry: entry[1].number or 0)


def resolve_experiment(
    selector: str | None, path: Path | None, directory: Path = CONFIG_DIRECTORY
) -> tuple[Path, ExperimentConfig]:
    if path is not None:
        if selector is not None:
            raise ValueError("Выберите номер исследования или --config, не оба сразу")
        target_path = path.parent if path.is_file() and path.name == STUDY_MANIFEST_FILE else path
        return target_path, load_experiment(target_path)
    if selector is None:
        default_path = directory / DEFAULT_CONFIG.name if directory != CONFIG_DIRECTORY else DEFAULT_CONFIG
        return default_path, load_experiment(default_path)
    matches = [
        (entry, config)
        for entry, config in catalogue(directory)
        if selector
        in {
            str(config.number),
            f"{config.number:02d}" if config.number is not None else "",
            entry.name if entry.is_dir() else entry.stem,
            config.name,
        }
    ]
    if len(matches) != 1:
        raise ValueError(f"Исследование {selector!r} не найдено. Каталог: uv run poe exp list")
    return matches[0]


def default_study(config: ExperimentConfig, split: str, pairs: list[str] | None, repeats: int) -> str:
    subset = "-".join(sorted({pair.lower() for pair in pairs})) if pairs else "all"
    return f"{config.name}-{split}-{subset}-r{repeats}"
