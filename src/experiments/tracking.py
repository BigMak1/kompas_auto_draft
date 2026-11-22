"""Локальная история MLflow: прогоны, версии агента, данные и переносимые артефакты."""

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from src.config import ROOT
from src.experiments.logging import log, log_error
from src.experiments.prompts import link_prompts
from src.experiments.registry import ArchivedInMlflow, ensure_datasets, ensure_experiment, ensure_model
from src.experiments.storage import runtime_database, study_lock
from src.experiments.store import read_record, verify_study
from src.utils import error_details, read_json, sha256

SYNC_VERSION = 3


def uri(root: Path) -> str:
    return "sqlite:///" + runtime_database(root).as_posix()


@contextmanager
def local_tracking(root: Path) -> Iterator[Any]:
    """Даже при внешнем MLFLOW_TRACKING_URI проект использует свою локальную базу."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    import mlflow

    previous = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    attachments = os.environ.get("MLFLOW_TRACE_EXTRACT_ATTACHMENTS")
    # Иначе MLflow выносит base64 в отдельный файл раньше фильтра span_processors.
    os.environ["MLFLOW_TRACE_EXTRACT_ATTACHMENTS"] = "false"
    mlflow.set_tracking_uri(uri(root))
    mlflow.set_registry_uri(uri(root))
    try:
        yield mlflow
    finally:
        mlflow.set_tracking_uri(previous)
        mlflow.set_registry_uri(previous_registry)
        if attachments is None:
            os.environ.pop("MLFLOW_TRACE_EXTRACT_ATTACHMENTS", None)
        else:
            os.environ["MLFLOW_TRACE_EXTRACT_ATTACHMENTS"] = attachments


def parameters(study: dict) -> dict:
    config = study["config"]
    result = {}
    for section in ("agent", "reading", "cad", "evaluation"):
        for key, value in config.get(section, {}).items():
            if key not in {"prompt", "passport_format", "notes_format"}:
                result[f"{section}.{key}"] = str(value)
    result["order_seed"] = str(config["order_seed"])
    for key in ("agent_version", "dataset_version"):
        if config.get(key):
            result[key] = config[key]
    for role, prompt in config.get("prompt_snapshot", {}).items():
        result[f"prompt.{role}"] = prompt["uri"]
    return result


def prepare_tracking(root: Path, directory: Path, *, restore_deleted: bool = False) -> dict:
    """Подготовка до генерации; код и эталоны не передаются в контекст LLM."""
    with local_tracking(root) as mlflow:
        from mlflow.tracking import MlflowClient

        study = verify_study(directory)
        config = study["config"]
        client = MlflowClient(tracking_uri=uri(root), registry_uri=uri(root))
        with FileLock(root / "mlflow/.runtime/catalog.lock"):
            experiment_id = ensure_experiment(
                client,
                config["name"] + " · " + config.get("title", "Сравнение маршрутов"),
                root / "mlflow/files" / config["name"],
                restore_deleted=restore_deleted,
                legacy_name="hw2-" + study["name"] if "number" not in config else None,
            )
            datasets = ensure_datasets(client, root, experiment_id, study, restore_deleted=restore_deleted)
            model_id = ensure_model(mlflow, client, directory, experiment_id, study)
        return {"study": study, "experiment_id": experiment_id, "model_id": model_id, **datasets}


def ensure_run(
    client: Any, root: Path, directory: Path, destination: Path, study: dict, experiment_id: str
) -> str:
    identity = destination.relative_to(directory).as_posix()
    found = client.search_runs(
        [experiment_id],
        filter_string=(f"tags.study = '{study['name']}' AND tags.attempt = '{identity}'"),
        max_results=2,
    )
    if found:
        return found[0].info.run_id
    # Совместимость с записями прежней реализации, до появления тега attempt.
    mapping_path = root / "mlflow/.runtime" / f"{study['name']}.json"
    mapping = read_json(mapping_path) if mapping_path.exists() else {}
    if identity in mapping:
        from mlflow.exceptions import MlflowException

        try:
            run = client.get_run(mapping[identity]["run_id"])
            if run.info.experiment_id == experiment_id:
                client.set_tag(run.info.run_id, "attempt", identity)
                return run.info.run_id
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
    record = read_record(destination)
    tags = {
        "mlflow.runName": identity,
        "study": study["name"],
        "attempt": identity,
        "split": study["split"],
        "kind": record["kind"],
        "pair_id": record["pair_id"],
    }
    return client.create_run(experiment_id, tags=tags).info.run_id


def sync_study(directory: Path, root: Path, *, restore_deleted: bool = False) -> dict:
    """Повторная синхронизация не дублирует результаты, метрики и связи."""
    context = prepare_tracking(root, directory, restore_deleted=restore_deleted)
    study = context["study"]
    from mlflow.entities import DatasetInput, InputTag, LoggedModelInput
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri(root), registry_uri=uri(root))
    for name in ("mlflow", "alembic", "sqlalchemy"):
        logging.getLogger(name).setLevel(logging.WARNING)
    count = measured = 0
    with FileLock(root / "mlflow/.runtime/sync.lock"):
        for path in sorted(directory.glob("*/*/record.json")):
            record = read_record(path.parent)
            identity = path.parent.relative_to(directory).as_posix()
            run_id = ensure_run(client, root, directory, path.parent, study, context["experiment_id"])
            saved = client.get_run(run_id)
            params = {
                **parameters(study),
                "variant": record.get("variant", record.get("stage", "reading")),
                "repeat": str(record["repeat"]),
                "model": study["config"]["agent"]["model"],
                "reasoning_effort": study["config"]["agent"]["reasoning_effort"],
                "max_output_tokens": str(study["config"]["agent"]["max_output_tokens"]),
                "timeout_seconds": str(study["config"]["agent"]["timeout_seconds"]),
                "study_fingerprint": study["fingerprint"],
                "agent_version_id": context["model_id"],
                "dataset_digest": context["input"].digest,
            }
            for key, value in params.items():
                if key not in saved.data.params:
                    client.log_param(run_id, key, value)
            client.set_tag(run_id, "outcome", record["status"])
            client.set_tag(run_id, "shared_reading", record.get("reading_path", ""))
            if record.get("error"):
                error = record["error"]
                client.set_tag(run_id, "error.stage", error.get("stage", record.get("stage", "unknown")))
                client.set_tag(run_id, "error.reason", error.get("summary", error["message"])[:4000])
            checksum = sha256(path)
            if saved.data.tags.get("record_sha256") != checksum:
                for key, value in {"seconds": record.get("seconds"), **(record.get("usage") or {})}.items():
                    if key in {"seconds", "input_tokens", "output_tokens"} and value is not None:
                        # Старые записи уже содержат эти точки — миграция не создаёт вторую.
                        if saved.data.metrics.get(key) != value:
                            client.log_metric(run_id, key, value)
                client.log_artifacts(run_id, str(path.parent))
                client.log_artifact(run_id, str(directory / "study.json"))
                client.set_tag(run_id, "record_sha256", checksum)
            if saved.data.tags.get("tracking_schema") != str(SYNC_VERSION):
                if (path.parent / "debug").is_dir():
                    client.log_artifacts(run_id, str(path.parent / "debug"), artifact_path="debug")
                client.log_inputs(
                    run_id,
                    datasets=[
                        DatasetInput(context["input"], [InputTag("mlflow.data.context", study["split"])])
                    ],
                    models=[LoggedModelInput(context["model_id"])],
                )
                client.set_tag(run_id, "dataset_snapshot", context["dataset_run_id"])
                roles = [record.get("stage", "passport")] if record["kind"] == "reading" else ["builder"]
                if record.get("variant") in {"passport_only", "passport_png", "notes_png"}:
                    roles.append("notes" if record["variant"] == "notes_png" else "passport")
                link_prompts(client, study, run_id=run_id, roles=roles)
                client.set_tag(run_id, "tracking_schema", str(SYNC_VERSION))
            score = root / "results" / study["name"] / identity / "evaluation.json"
            if score.is_file():
                evaluation = read_json(score)
                if sha256(score) != score.with_suffix(".sha256").read_text().strip():
                    raise ValueError("Сохранённая оценка изменена")
                if (
                    evaluation.get("record_sha256") != checksum
                    or evaluation.get("study_fingerprint") != study["fingerprint"]
                ):
                    raise ValueError("Оценка относится к другой попытке или серии")
                if any(evaluation.get(key) is not None for key in ("iou", "f_score")) and (
                    evaluation.get("evaluator_sha256") != study["source_sha256"]["src/evaluate.py"]
                ):
                    raise ValueError("Метрики рассчитаны другой версией оценщика")
                if saved.data.tags.get("evaluation_sha256") != sha256(score):
                    for key in ("iou", "f_score"):
                        if evaluation.get(key) is not None:
                            client.log_metric(
                                run_id,
                                key,
                                evaluation[key],
                                model_id=context["model_id"],
                                dataset_name=context["input"].name,
                                dataset_digest=context["input"].digest,
                            )
                    client.log_artifact(run_id, str(score))
                    client.set_tag(run_id, "evaluation_sha256", sha256(score))
                measured += int(evaluation.get("iou") is not None and evaluation.get("f_score") is not None)
            if record["status"] != "started":
                status = "FINISHED" if record["status"] == "success" else "FAILED"
                if saved.info.status != status:
                    client.set_terminated(run_id, status=status)
            count += 1
    log(
        "MLFLOW",
        f"{study['name']}: записей {count}, оценок {measured}. Эксперимент №{context['experiment_id']}.",
    )
    return context


def sync_all(root: Path) -> bool:
    success = True
    for path in sorted((root / "run").glob("*/study.json")):
        lock = FileLock(study_lock(root, path.parent.name), timeout=0)
        try:
            lock.acquire()
        except Timeout:
            log("ПРОПУСК", f"{path.parent.name}: серия выполняется.")
            continue
        try:
            sync_study(path.parent, root)
        except ArchivedInMlflow as error:
            log("ПРОПУСК", f"{path.parent.name}: {error}.")
        except Exception as error:  # noqa: BLE001 — ошибка одной серии не скрывает остальные.
            success = False
            log_error(error_details(error, "синхронизация MLflow"), root / "logs/tracking/sync-error.log")
        finally:
            lock.release()
    return success


def main(argv: list[str] | None = None) -> int:
    import argparse
    import subprocess

    from src.experiments.storage import publish, restore

    parser = argparse.ArgumentParser(description="Локальная история MLflow и публикация переносимого снимка")
    parser.add_argument("command", nargs="?", choices=("publish", "restore"))
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)
    try:
        if args.command == "publish":
            if not sync_all(ROOT):
                raise ValueError("Синхронизация не завершена; снимок не опубликован")
            snapshot = publish(ROOT)
            log("MLFLOW", f"Снимок опубликован локально: {len(snapshot['files'])} файлов. Git не изменялся.")
            return 0
        if args.command == "restore":
            with FileLock(ROOT / "logs/.locks/mlflow-server.lock", timeout=0):
                restore(ROOT)
            log("MLFLOW", "Рабочая история архивирована; опубликованный снимок восстановлен.")
            return 0
        database = uri(ROOT)
        sync_all(ROOT)
        with FileLock(study_lock(ROOT, "mlflow-server"), timeout=0):
            log(
                "MLFLOW",
                f"Открывай http://127.0.0.1:{args.port} · Runs, Traces, Datasets, Prompts, Agent versions.",
            )
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "mlflow",
                    "server",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--backend-store-uri",
                    database,
                    "--no-serve-artifacts",
                ],
                cwd=ROOT,
            )
    except Exception as error:  # noqa: BLE001 — ошибка команды сохраняется без потери истории.
        log_error(error_details(error, "MLflow"), ROOT / "logs/tracking/command-error.log")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
