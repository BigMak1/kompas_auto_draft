"""Проверка входов без импорта KsAPI и без открытия КОМПАС."""

from pathlib import Path
from typing import Any

from eval.geometry.io import read_json, sha256
from eval.geometry.models import Case, Evaluation, Facts, ModelInput

REPO = Path(__file__).resolve().parents[2]


def worker_fingerprint() -> dict[str, str]:
    return {
        "worker_sha256": sha256(Path(__file__).with_name("kompas_worker.py")),
        "vendor_sha256": sha256(REPO / "vendor/kompas_lib/kompas_lib.py"),
        "normalizer_sha256": sha256(Path(__file__).with_name("normalize.py")),
    }


def model_issues(
    model: ModelInput, root: Path, evaluation: Evaluation, *, registration: bool
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    def problem(code: str, path: str, message: str) -> None:
        issues.append({"code": code, "path": path, "message": message})

    for field in ("mesh", "native", "facts", "edge_points", "export_manifest"):
        value = getattr(model, field)
        if value and not (root / value).is_file():
            problem("missing_file", value, f"Нет файла {field}: {value}")
    if registration and not model.facts:
        problem("missing_facts", "", "Для rigid нужен снимок Facts из КОМПАС")
    facts = None
    if model.facts and (root / model.facts).is_file():
        try:
            facts = Facts.model_validate(read_json(root / model.facts))
            if facts.source != "kompas" and evaluation.purpose != "metric_validation":
                raise ValueError("Аналитические факты недопустимы для benchmark")
            if facts.mesh_units != model.units:
                raise ValueError("Единицы сетки и Facts не совпадают")
            if (root / model.mesh).is_file() and facts.mesh_sha256 != sha256(root / model.mesh):
                raise ValueError("Хеш сетки не совпадает с Facts")
            if model.native and (root / model.native).is_file():
                if facts.native_sha256 != sha256(root / model.native):
                    raise ValueError("Хеш M3D не совпадает с Facts")
            if (
                registration
                and facts.solid_valid is not False
                and any(
                    item is None for item in (facts.center_mm, facts.principal_axes, facts.principal_moments)
                )
            ):
                raise ValueError("Для rigid нужны center_mm, principal_axes, principal_moments")
        except (ValueError, OSError) as error:
            problem("invalid_facts", model.facts, str(error))
    if model.export_manifest and (root / model.export_manifest).is_file():
        try:
            record = read_json(root / model.export_manifest)
            if record["profile"] != evaluation.export.model_dump():
                raise ValueError("Параметры сохранённой выгрузки устарели")
            if record["implementation"] != worker_fingerprint():
                raise ValueError("Версия экспортёра изменилась; нужна новая выгрузка")
            directory = (root / model.export_manifest).parent
            for name, expected in record["files"].items():
                if sha256(directory / name) != expected:
                    raise ValueError(f"Хеш артефакта выгрузки не совпал: {name}")
            if facts is None or record["native_sha256"] != facts.native_sha256:
                raise ValueError("Манифест выгрузки не соответствует Facts")
            if record["files"].get("model.stl") != sha256(root / model.mesh):
                raise ValueError("Манифест выгрузки не соответствует сетке")
            if record["files"].get("eval-facts.json") != sha256(root / model.facts):
                raise ValueError("Манифест выгрузки не соответствует файлу Facts")
        except (ValueError, OSError, KeyError, TypeError) as error:
            problem("stale_export", model.export_manifest, str(error))
    return issues


def case_issues(case: Case, root: Path, evaluation: Evaluation) -> list[dict[str, str]]:
    issues = []
    roles = [("reference", case.reference)]
    if case.generation_status == "success" and case.prediction:
        roles.append(("prediction", case.prediction))
    for role, model in roles:
        for item in model_issues(
            model,
            root,
            evaluation,
            registration=evaluation.config.alignment == "rigid" and case.generation_status == "success",
        ):
            issues.append({"case_id": case.id, "role": role, **item})
    return issues


def inspect_inputs(root: Path, evaluation: Evaluation) -> dict[str, Any]:
    issues = [item for case in evaluation.cases for item in case_issues(case, root, evaluation)]
    return {"ready_for_surface": not issues, "issues": issues}
