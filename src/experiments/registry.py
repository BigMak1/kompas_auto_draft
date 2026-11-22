"""Неизменяемые версии набора данных и агента поверх штатных сущностей MLflow."""

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from src.experiments.logging import log
from src.experiments.prompts import link_prompts
from src.experiments.store import digest
from src.utils import read_json, sha256


class ArchivedInMlflow(ValueError):
    """Фоновая синхронизация пропускает записи, перемещённые в корзину."""


def ensure_experiment(
    client: Any, name: str, directory: Path, *, restore_deleted: bool, legacy_name: str | None = None
) -> str:
    from mlflow.entities import LifecycleStage

    experiment = client.get_experiment_by_name(legacy_name) if legacy_name else None
    experiment = experiment or client.get_experiment_by_name(name)
    if experiment is None:
        return client.create_experiment(name, artifact_location=directory.resolve().as_uri())
    if experiment.lifecycle_stage == LifecycleStage.DELETED:
        if not restore_deleted:
            raise ArchivedInMlflow(
                f"Эксперимент MLflow «{experiment.name}» (№{experiment.experiment_id}) в корзине"
            )
        client.restore_experiment(experiment.experiment_id)
        log("MLFLOW", f"Восстановлен эксперимент «{experiment.name}» (№{experiment.experiment_id}).")
    return experiment.experiment_id


def local_artifact_path(location: str, root: Path) -> Path:
    parsed = urlparse(location)
    if parsed.scheme not in ("file", ""):
        raise ValueError("Для публикации поддерживаются локальные артефакты MLflow")
    path = Path(unquote(parsed.path))
    path = path if path.is_absolute() else root / path
    if not path.resolve().is_relative_to((root / "mlflow/files").resolve()):
        raise ValueError("Артефакт MLflow находится вне mlflow/files")
    return path


def verify_dataset_files(directory: Path, checksum: str) -> None:
    manifest = read_json(directory / "manifest.json")
    if dataset_fingerprint(manifest) != checksum:
        raise ValueError("Изменён манифест опубликованной версии датасета")
    for item in manifest["items"]:
        for key in ("input_png", "target_mesh"):
            asset = item[key]
            path = directory / asset["path"]
            if (
                not path.resolve().is_relative_to(directory.resolve())
                or path.is_symlink()
                or not path.is_file()
                or sha256(path) != asset["sha256"]
            ):
                raise ValueError(f"Изменён артефакт версии датасета: {asset['path']}")


def dataset_records(manifest: dict, split: str, run_id: str) -> list[dict]:
    return [
        {
            "inputs": {
                "pair_id": item["pair_id"],
                "png": f"runs:/{run_id}/{item['input_png']['path']}",
                "png_sha256": item["input_png"]["sha256"],
                "orientation": item["orientation"],
            },
            "expectations": {
                "stl": f"runs:/{run_id}/{item['target_mesh']['path']}",
                "stl_sha256": item["target_mesh"]["sha256"],
                "family": item.get("family"),
                "limitations": item.get("limitations", []),
                "review": item.get("review", {}),
            },
        }
        for item in manifest["items"]
        if item["split"] == split
    ]


def dataset_fingerprint(manifest: dict) -> str:
    # Старое имя cad-hw2-* и новое drawing-cad-* обозначали одни файлы и разметку.
    # Имя набора — метка; все содержательные поля манифеста входят в отпечаток.
    return digest({key: value for key, value in manifest.items() if key != "dataset_id"})


def ensure_datasets(
    client: Any, root: Path, experiment_id: str, study: dict, *, restore_deleted: bool
) -> dict[str, Any]:
    from mlflow.entities import Dataset, LifecycleStage, ViewType
    from mlflow.genai.datasets import create_dataset, search_datasets

    manifest = study["manifest"]
    checksum = dataset_fingerprint(manifest)
    version = study["config"].get("dataset_version") or "drawing-cad-v3"
    catalog_id = ensure_experiment(
        client,
        "Datasets · Версии чертежей и CAD",
        root / "mlflow/files/datasets",
        restore_deleted=restore_deleted,
    )
    found = client.search_runs(
        [catalog_id],
        filter_string=f"tags.dataset_version = '{version}'",
        run_view_type=ViewType.ALL,
        max_results=2,
    )
    if len(found) > 1:
        raise ValueError(f"Дубли версии датасета {version}")
    if found and found[0].data.tags.get("content_sha256") != checksum:
        raise ValueError(f"Датасет {version} изменён: укажите новую dataset_version")
    if found:
        run_id = found[0].info.run_id
        if found[0].data.tags.get("snapshot_complete") != "true":
            raise ValueError(f"Снимок датасета {version} не завершён; требуется восстановление каталога")
        verify_dataset_files(local_artifact_path(found[0].info.artifact_uri, root), checksum)
        if found[0].info.lifecycle_stage == LifecycleStage.DELETED:
            if not restore_deleted:
                raise ArchivedInMlflow(f"Снимок датасета {version} в корзине MLflow")
            client.restore_run(run_id)
            log("MLFLOW", f"Восстановлен снимок датасета {version} с ID {run_id}.")
    else:
        run_id = client.create_run(
            catalog_id,
            tags={
                "mlflow.runName": version,
                "dataset_version": version,
                "content_sha256": checksum,
            },
        ).info.run_id
        # В отдельной записи хранится весь набор, один раз для всех исследований.
        for item in manifest["items"]:
            for key in ("input_png", "target_mesh"):
                asset = item[key]
                path = (root / study["config"]["manifest"]).parent / asset["path"]
                if not path.is_file() or path.is_symlink() or sha256(path) != asset["sha256"]:
                    raise ValueError(f"Изменён файл датасета: {asset['path']}")
                client.log_artifact(run_id, str(path), artifact_path=Path(asset["path"]).parent.as_posix())
        client.log_dict(run_id, manifest, "manifest.json")
        client.set_tag(run_id, "snapshot_complete", "true")
        client.set_terminated(run_id)
    existing = search_datasets(max_results=1000)
    for split in sorted({item["split"] for item in manifest["items"]}):
        name = f"{version}-{split}"
        matching = [d for d in existing if d.name == name]
        # В истории есть одноимённые карточки. Для нового эксперимента берём карточку каталога.
        dataset = max(
            matching,
            key=lambda d: (experiment_id in d.experiment_ids, catalog_id in d.experiment_ids),
            default=None,
        )
        records = dataset_records(manifest, split, run_id)
        if dataset:
            if dataset.tags.get("content_sha256") != checksum:
                raise ValueError(f"Карточка {name} относится к другому содержимому")
            rows = dataset.to_df().to_dict(orient="records")
            actual = {r["inputs"]["pair_id"]: {k: r[k] for k in ("inputs", "expectations")} for r in rows}
            if len(actual) != len(rows) or actual != {r["inputs"]["pair_id"]: r for r in records}:
                raise ValueError(f"Карточка {name} изменена: создайте новую версию датасета")
            linked = {eid for card in matching for eid in card.experiment_ids}
            missing = sorted({experiment_id, catalog_id} - linked)
            if missing:
                client.add_dataset_to_experiments(dataset.dataset_id, missing)
        else:
            dataset = create_dataset(
                name=name,
                experiment_id=[experiment_id, catalog_id],
                tags={
                    "split": split,
                    "version": version,
                    "content_sha256": checksum,
                    "snapshot_run_id": run_id,
                },
            )
            dataset.merge_records(records)
    selected = [i for i in manifest["items"] if i["pair_id"] in study["pairs"]]
    return {
        "dataset_run_id": run_id,
        "input": Dataset(
            name=version + "-" + study["split"],
            digest=digest(selected)[:32],
            source_type="code",
            source=json.dumps({"tags": {"manifest": f"runs:/{run_id}/manifest.json", "version": version}}),
            profile=json.dumps({"num_rows": len(selected)}),
        ),
    }


def agent_fingerprint(study: dict) -> str:
    config = study["config"]
    identity = {
        "agent": config["agent"],
        "reading": config["reading"],
        "cad": config["cad"],
        "source": study["source_sha256"],
        "dependencies": study["versions"],
    }
    if config.get("prompt_snapshot"):
        identity["prompts"] = config["prompt_snapshot"]
    return digest(identity)


def ensure_model(mlflow: Any, client: Any, directory: Path, experiment_id: str, study: dict) -> str:
    from mlflow.entities import ViewType

    from src.experiments.tracking import parameters

    config = study["config"]
    checksum = agent_fingerprint(study)
    version = config.get("agent_version")
    experiments = [e.experiment_id for e in client.search_experiments(view_type=ViewType.ALL)]
    key, value = ("agent_version", version) if version else ("agent_fingerprint", checksum)
    found = client.search_logged_models(experiments, filter_string=f"tags.{key} = '{value}'", max_results=2)
    if not found and not version:
        found = client.search_logged_models(
            experiments, filter_string=f"tags.agent_version = '{checksum}'", max_results=2
        )
    if found:
        if len(found) != 1:
            raise ValueError(f"Дубли версии агента {version}")
        saved = found[0].tags.get("agent_fingerprint", found[0].tags.get("agent_version"))
        if saved != checksum:
            raise ValueError(f"Агент {version} изменён: укажите новую agent_version")
        return found[0].model_id
    label = version or f"cad-agent-legacy-{checksum[:12]}"
    model = mlflow.create_external_model(
        name=label,
        model_type="agent",
        experiment_id=experiment_id,
        tags={
            "agent_version": label,
            "agent_fingerprint": checksum,
            "config_version": config["name"],
            "description": "Версия кода, зависимостей, модели, схем, промптов и настроек CAD",
        },
        params={k: v for k, v in parameters(study).items() if not k.startswith("evaluation.")},
    )
    client.log_model_artifact(model.model_id, str(directory / "study.json"))
    client.log_model_artifact(model.model_id, str(directory / "source.zip"))
    link_prompts(client, study, model_id=model.model_id)
    return model.model_id


def verify_registry(root: Path, client: Any) -> None:
    """Перед публикацией проверяем и старые версии, даже если их portable-run уже удалён."""
    from mlflow.entities import ViewType

    experiments = [e.experiment_id for e in client.search_experiments(view_type=ViewType.ALL)]
    for model in client.search_logged_models(experiments, max_results=1000):
        directory = local_artifact_path(model.artifact_location, root)
        study = read_json(directory / "study.json")
        expected = model.tags.get("agent_fingerprint", model.tags.get("agent_version"))
        if (
            agent_fingerprint(study) != expected
            or sha256(directory / "source.zip") != study["source_archive_sha256"]
        ):
            raise ValueError(f"Изменён снимок версии агента {model.name}")
    for run in client.search_runs(
        experiments, filter_string="tags.snapshot_complete = 'true'", max_results=1000
    ):
        verify_dataset_files(
            local_artifact_path(run.info.artifact_uri, root), run.data.tags["content_sha256"]
        )
