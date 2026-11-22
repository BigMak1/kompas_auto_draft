"""Закреплённые версии Prompt Registry; снимок текстов автономен от MLflow."""

import hashlib
from pathlib import Path
from typing import Any

from src.experiments.config import ExperimentConfig

NAMES = {"builder": "cad-builder", "passport": "drawing-passport", "notes": "drawing-notes"}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def legacy_texts(config: dict) -> dict[str, str]:
    reading = config["reading"]
    return {
        "builder": config["agent"]["prompt"],
        "passport": reading["prompt"] + "\n" + reading["passport_format"],
        "notes": reading["prompt"] + "\n" + reading["notes_format"],
    }


def register_texts(client: Any, texts: dict[str, str], message: str) -> dict[str, str]:
    """Первичное заполнение/миграция: одинаковый текст не создаёт новую версию."""
    from mlflow.exceptions import MlflowException

    refs = {}
    for role, text in texts.items():
        name = NAMES[role]
        try:
            versions = client.search_prompt_versions(name, max_results=1000)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            versions = []
        found = next((v for v in versions if v.template == text), None)
        prompt = found or client.register_prompt(
            name, text, commit_message=message, tags={"sha256": text_hash(text)}
        )
        refs[role] = f"prompts:/{name}/{prompt.version}"
    return refs


def resolve_prompts(root: Path, config: ExperimentConfig) -> ExperimentConfig:
    """Разрешаем строго номерные ссылки до генерации; никакого fallback на latest или YAML."""
    if config.prompts is None:
        return config
    from mlflow.tracking import MlflowClient

    from src.experiments.tracking import local_tracking, uri

    with local_tracking(root):
        client = MlflowClient(tracking_uri=uri(root), registry_uri=uri(root))
        snapshots = {}
        for role, reference in config.prompts.model_dump().items():
            prompt = client.load_prompt(reference, cache_ttl_seconds=0)
            if prompt is None or not isinstance(prompt.template, str) or not prompt.template.strip():
                raise ValueError(f"Не найден текст закреплённого промпта: {reference}")
            checksum = text_hash(prompt.template)
            registered = prompt.tags.get("sha256")
            if registered is not None and registered != checksum:
                raise ValueError(f"Изменено содержимое версии промпта: {reference}")
            snapshots[role] = {
                "uri": reference,
                "version": int(prompt.version),
                "text": prompt.template,
                "sha256": checksum,
            }
    if config.prompt_snapshot and config.prompt_snapshot != snapshots:
        raise ValueError("Закреплённые промпты не совпадают со снимком серии")
    return config.model_copy(
        update={
            "prompt_snapshot": snapshots,
            "agent": config.agent.model_copy(
                update={"prompt": snapshots["builder"]["text"], "prompt_version": snapshots["builder"]["uri"]}
            ),
            "reading": config.reading.model_copy(
                update={
                    "prompt": "",
                    "prompt_version": snapshots["passport"]["uri"],
                    "passport_format": snapshots["passport"]["text"],
                    "notes_format": snapshots["notes"]["text"],
                }
            ),
        }
    )


def link_prompts(
    client: Any,
    study: dict,
    *,
    run_id: str | None = None,
    model_id: str | None = None,
    trace_id: str | None = None,
    roles: list[str] | None = None,
) -> None:
    snapshots = study["config"].get("prompt_snapshot", {})
    versions = []
    for role, snapshot in snapshots.items():
        if roles is not None and role not in roles:
            continue
        ref = snapshot["uri"]
        if run_id:
            client.link_prompt_version_to_run(run_id, ref)
        if model_id:
            client.link_prompt_version_to_model(NAMES[role], str(snapshot["version"]), model_id)
        if trace_id:
            versions.append(client.load_prompt(ref, cache_ttl_seconds=0))
    if versions:
        client.link_prompt_versions_to_trace(versions, trace_id)
