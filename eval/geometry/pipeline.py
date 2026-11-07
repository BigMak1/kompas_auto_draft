"""Подготовка → офлайн-оценка → пересечение → итог. KsAPI здесь не импортируется."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval.geometry.backend import BackendSettings, WorkerClient
from eval.geometry.io import atomic_json, create_artifact_directory, read_json, sha256
from eval.geometry.models import BooleanResult, Evaluation, ModelInput
from eval.geometry.normalize import normalize_export
from eval.geometry.progress import ProgressCallback, no_progress
from eval.geometry.readiness import inspect_inputs, model_issues
from eval.geometry.runner import finalize_solid_results, load_model, run_evaluation


def load_evaluation(root: Path) -> Evaluation:
    original = Evaluation.model_validate(read_json(root / "evaluation.json"))
    state_path = root / "evaluation/state.json"
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("input_sha256") == sha256(root / "evaluation.json"):
            return Evaluation.model_validate(state["evaluation"])
    return original


def save_state(root: Path, evaluation: Evaluation, input_hash: str) -> None:
    if sha256(root / "evaluation.json") != input_hash:
        raise ValueError("evaluation.json изменился во время подготовки; сохранённые артефакты оставлены")
    atomic_json(
        root / "evaluation/state.json",
        {
            "input_sha256": input_hash,
            "evaluation": evaluation.model_dump(),
        },
    )


def execute(
    root: Path,
    *,
    offline: bool = False,
    backend: BackendSettings | None = None,
    prepare_only: bool = False,
    progress: ProgressCallback = no_progress,
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    progress("Проверка входов и сохранённых выгрузок")
    root = root.resolve()
    input_hash = sha256(root / "evaluation.json")
    evaluation = load_evaluation(root)
    started_at = datetime.now().astimezone()
    session = create_artifact_directory(root / "evaluation", "pipeline", started_at)
    client = WorkerClient(backend, session / "worker") if backend and not offline else None
    preparation: dict[str, Any] = {
        "started_at": started_at.isoformat(),
        "mode": "offline" if offline else "auto",
        "before": inspect_inputs(root, evaluation),
        "actions": [],
        "backend": "not_needed",
    }
    existing: dict[str, dict] = {}
    unavailable_results: str | None = None
    cache_event: dict[str, Any] | None = None
    if evaluation.boolean_results:
        try:
            for data in read_json(root / evaluation.boolean_results):
                item = BooleanResult.model_validate(data)
                if item.request_sha256 in existing:
                    raise ValueError("Повтор результата пересечения")
                existing[item.request_sha256] = item.model_dump()
        except (ValueError, OSError, TypeError) as error:
            cache_event = {
                "boolean_results": evaluation.boolean_results,
                "warning": "Сохранённый файл пересечений недоступен",
                "reason": str(error),
                "status": "unavailable",
            }
            preparation["actions"].append(cache_event)
            progress("Файл пересечений недоступен; потребуются новые ответы КОМПАС для IoU")
            existing = {}
            unavailable_results = evaluation.boolean_results
            evaluation = evaluation.model_copy(update={"boolean_results": None})

    def usable_client() -> bool:
        if client is None:
            preparation["backend"] = "offline" if offline else "not_configured"
            return False
        if client.available is None:
            try:
                progress("Ожидание ответа КОМПАС: проверка подключения")
                preparation["health"] = client.health()
            except RuntimeError as error:
                preparation["backend_error"] = str(error)
        preparation["backend"] = "available" if client.available else "unavailable"
        return bool(client.available)

    cases = []
    for case in evaluation.cases:
        replacements = {}
        for role in ("reference", "prediction"):
            model = getattr(case, role)
            if model is None or (role == "prediction" and case.generation_status != "success"):
                continue
            issues = model_issues(
                model,
                root,
                evaluation,
                registration=evaluation.config.alignment == "rigid" and case.generation_status == "success",
            )
            if not issues or not model.native or not (root / model.native).is_file():
                continue
            if not usable_client():
                continue
            try:
                label = "эталон" if role == "reference" else "предсказание"
                progress(f"{case.id} · Выгрузка из КОМПАС: {label}")
                job, _ = client.invoke(
                    "export",
                    {"profile": evaluation.export.model_dump()},
                    {
                        "model": root / model.native,
                    },
                )
                normalize_export(job)
                prepared = ModelInput(
                    mesh=str((job / "model.stl").relative_to(root)),
                    units="mm",
                    native=model.native,
                    facts=str((job / "eval-facts.json").relative_to(root)),
                    export_manifest=str((job / "export.json").relative_to(root)),
                    # Старые edge_points относятся к старой выгрузке и автоматически не переносятся.
                )
                remaining = model_issues(prepared, root, evaluation, registration=True)
                if remaining:
                    raise ValueError("; ".join(item["message"] for item in remaining))
                # Проверка STL библиотекой, а не только кодом возврата конвертера.
                load_model(prepared, root, evaluation.purpose == "metric_validation")
                replacements[role] = prepared
                preparation["actions"].append({"case_id": case.id, "role": role, "export": str(job)})
            except (OSError, RuntimeError, ValueError) as error:
                preparation["actions"].append({"case_id": case.id, "role": role, "error": str(error)})
        cases.append(case.model_copy(update=replacements))
    evaluation = evaluation.model_copy(update={"cases": cases})
    preparation["after"] = inspect_inputs(root, evaluation)
    atomic_json(session / "preparation.json", preparation)
    atomic_json(session / "evaluation.json", evaluation.model_dump())
    save_state(
        root,
        evaluation.model_copy(update={"boolean_results": unavailable_results})
        if unavailable_results
        else evaluation,
        input_hash,
    )
    if prepare_only:
        preparation["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(session / "preparation.json", preparation)
        return session, preparation
    if not preparation["actions"] and preparation["after"]["ready_for_surface"]:
        progress("Готовые сетки и измерения использованы повторно, без выгрузки")
    output = run_evaluation(root, evaluation=evaluation, progress=progress)
    preliminary = output
    requests = read_json(output / "raw/solid_requests.json")
    updated = False
    if requests and all(existing.get(r["request_sha256"], {}).get("status") == "ok" for r in requests):
        progress("IoU: используются сохранённые результаты пересечений")
    for request in requests:
        if existing.get(request["request_sha256"], {}).get("status") == "ok":
            continue
        if not usable_client():
            break
        case = next(case for case in evaluation.cases if case.id == request["case_id"])
        if not case.prediction.native or not case.reference.native:
            continue
        try:
            pose = "после совмещения" if request["pose"] == "aligned" else "исходная поза"
            progress(f"{case.id} · Ожидание КОМПАС: пересечение тел ({pose})")
            mesh, _, _ = load_model(case.prediction, root, evaluation.purpose == "metric_validation")
            mesh.apply_transform(np.array(request["prediction_to_reference"]))
            job, response = client.invoke(
                "intersect",
                {
                    "solid_request": request,
                    "prediction_bounds_mm": mesh.bounds.reshape(-1).tolist(),
                    "bbox_tolerance_mm": 2 * evaluation.export.chord_mm,
                },
                {"prediction": root / case.prediction.native, "reference": root / case.reference.native},
            )
            result = BooleanResult.model_validate(response["result"])
            if result.request_sha256 != request["request_sha256"]:
                raise ValueError("Вычислитель ответил на другой запрос")
            existing[result.request_sha256] = result.model_dump()
            preparation["actions"].append({"intersection": str(job), "request_sha256": result.request_sha256})
            updated = True
        except (OSError, RuntimeError, ValueError) as error:
            preparation["actions"].append({"request_sha256": request["request_sha256"], "error": str(error)})
    if updated:
        results_path = session / "boolean-results.json"
        atomic_json(results_path, list(existing.values()))
        evaluation = evaluation.model_copy(update={"boolean_results": str(results_path.relative_to(root))})
        save_state(root, evaluation, input_hash)
        # Первый проход хранится как самостоятельное измерение; входы не перезаписываются.
        output = finalize_solid_results(root, preliminary, evaluation, progress)
    if cache_event is not None:
        if not requests:
            cache_event["status"] = "not_needed"
        elif all(existing.get(r["request_sha256"], {}).get("status") == "ok" for r in requests):
            cache_event["status"] = "restored"
            cache_event["replacement"] = evaluation.boolean_results
    progress("Сохранение результатов оценки")
    preparation["result"] = str(output)
    preparation["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(session / "preparation.json", preparation)
    atomic_json(output / "raw/preparation.json", preparation)
    manifest = read_json(output / "manifest.json")
    manifest["cost"]["total_seconds"] = preparation["elapsed_seconds"]
    atomic_json(output / "manifest.json", manifest)
    return output, preparation
