"""Офлайн-регрессии версий, локального MLflow и переноса опубликованного снимка."""

import copy
import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from filelock import FileLock, Timeout
from pydantic import ValidationError
from test_experiment import PASSPORT, Model, run
from test_experiment import sandbox as sandbox  # общие маленькие PNG/STL и подставленный ответ

from src.agent.schemas.passport import Passport
from src.agent.stages import diagnose_output
from src.config import ROOT
from src.experiments.config import ExperimentConfig, PromptVersions
from src.experiments.prompts import legacy_texts, register_texts, resolve_prompts
from src.experiments.storage import publish, restore, runtime_database, study_lock, verify_snapshot
from src.experiments.store import freeze, read_record, verify_study
from src.experiments.tracking import local_tracking, prepare_tracking, sync_all, uri
from src.utils import read_json, sha256, write_json


def registered(sandbox: tuple[Path, ExperimentConfig]) -> tuple[Path, ExperimentConfig]:
    from mlflow import MlflowClient

    root, config = sandbox
    with local_tracking(root):
        client = MlflowClient(tracking_uri=uri(root), registry_uri=uri(root))
        refs = register_texts(client, legacy_texts(config.model_dump()), "Офлайн-пример")
        assert refs == register_texts(client, legacy_texts(config.model_dump()), "Без дублей")
    return root, config.model_copy(update={"prompts": PromptVersions(**refs)})


def test_actual_saved_response_explains_both_errors_without_repair() -> None:
    path = ROOT / "tests/fixtures/passport-response-v1.json"
    checksum = sha256(path)
    response = read_json(path)
    output = json.loads(response["output"][-1]["content"][0]["text"])
    # Только тестовая копия проверяется новой схемой; исходный ответ не меняется.
    output.update(schema_version="drawing-passport-2", general_notes=[])
    adapted = copy.deepcopy(response)
    adapted["output"][-1]["content"][0]["text"] = json.dumps(output)
    error = diagnose_output(adapted, Passport)
    assert error["code"] == "schema_validation"
    fields = {tuple(v["loc"]): v["msg"] for v in error["validation"]}
    assert "dimensionless" in fields[("dimensions", 34, "unit")]
    assert "general_notes" in fields[("dimensions", 45, "feature_id")]
    output["dimensions"][34]["unit"] = "dimensionless"
    general = output["dimensions"].pop(45)
    output["general_notes"] = [{"raw": general["raw"], "view_id": general["view_id"], "status": "observed"}]
    Passport.model_validate(output)
    assert sha256(path) == checksum


@pytest.mark.parametrize(
    "change,field",
    [
        ({"unit": None}, "unit"),
        ({"unit": "kg"}, "unit"),
        ({"reading_status": "unresolved"}, "value"),
        ({"feature_id": None}, "feature_id"),
        ({"binding_status": "unresolved"}, "feature_id"),
    ],
)
def test_passport_field_errors(change: dict, field: str) -> None:
    value = copy.deepcopy(PASSPORT)
    value["dimensions"][0].update(change)
    with pytest.raises(ValidationError) as caught:
        Passport.model_validate(value)
    assert any(e["loc"] == ("dimensions", 0, field) for e in caught.value.errors())


def test_general_note_reference_is_checked() -> None:
    value = {**PASSPORT, "general_notes": [{"raw": "H14, h14", "view_id": "absent", "status": "observed"}]}
    with pytest.raises(ValidationError) as caught:
        Passport.model_validate(value)
    assert caught.value.errors()[0]["loc"] == ("general_notes", 0, "view_id")
    assert caught.value.errors()[0]["type"] == "passport_reference"


def test_active_study_is_skipped_but_real_sync_failure_is_logged(sandbox, monkeypatch, capsys) -> None:
    from src.experiments import tracking

    root, _ = sandbox
    run(sandbox, Model())
    with FileLock(study_lock(root, "unit")):
        assert sync_all(root)
    assert "серия выполняется" in capsys.readouterr().out
    assert not (root / "logs/tracking/sync-error.log").exists()

    def broken(*args):
        raise ValueError("Настоящая ошибка")

    monkeypatch.setattr(tracking, "sync_study", broken)
    assert not sync_all(root)
    assert "Настоящая ошибка" in (root / "logs/tracking/sync-error.log").read_text()


@pytest.mark.parametrize("delete_experiments", [True, False], ids=["experiment-and-catalog", "snapshot-only"])
def test_run_reuses_archived_mlflow_ids_but_background_sync_skips_them(sandbox, delete_experiments) -> None:
    from mlflow.entities import ViewType

    from mlflow import MlflowClient

    root, config = registered(sandbox)
    directory, _ = freeze(
        root, "unit", resolve_prompts(root, config), read_json(root / config.manifest), "dev", None, 1
    )
    prepared = prepare_tracking(root, directory)
    client = MlflowClient(tracking_uri=uri(root), registry_uri=uri(root))
    snapshot_id = prepared["dataset_run_id"]
    catalog_id = client.get_run(snapshot_id).info.experiment_id
    experiment_id = prepared["experiment_id"]
    other = client.create_experiment("Несвязанное исследование")
    client.delete_experiment(other)
    if delete_experiments:
        client.delete_experiment(experiment_id)
        client.delete_experiment(catalog_id)
    else:
        client.delete_run(snapshot_id)

    assert sync_all(root)
    assert not (root / "logs/tracking/sync-error.log").exists()
    assert client.get_run(snapshot_id).info.lifecycle_stage == "deleted"
    if delete_experiments:
        assert client.get_experiment(experiment_id).lifecycle_stage == "deleted"
        assert client.get_experiment(catalog_id).lifecycle_stage == "deleted"

    model = Model()
    run((root, config), model, tracking=True)
    assert len(model.calls) == 6
    assert {r.info.run_id for r in client.search_runs([catalog_id], run_view_type=ViewType.ALL)} == {
        snapshot_id
    }
    assert client.get_run(snapshot_id).info.lifecycle_stage == "active"
    assert client.get_experiment(other).lifecycle_stage == "deleted"
    assert {m.model_id for m in client.search_logged_models([experiment_id])} == {prepared["model_id"]}
    runs = client.search_runs([experiment_id])
    assert len(runs) == 6 and all(r.data.tags.get("dataset_snapshot") == snapshot_id for r in runs)
    assert all(r.data.tags["outcome"] == "success" for r in runs)


def test_pinned_prompts_are_snapshotted_and_linked(sandbox) -> None:
    from mlflow import MlflowClient

    root, config = registered(sandbox)
    directory = run((root, config), Model(), tracking=True)
    study = verify_study(directory)
    assert study["config"]["prompt_snapshot"]["builder"]["text"] == config.agent.prompt
    with local_tracking(root):
        client = MlflowClient(tracking_uri=uri(root), registry_uri=uri(root))
        experiment = client.get_experiment_by_name(config.name + " · " + config.title)
        model = client.search_logged_models([experiment.experiment_id])[0]
        assert model.name == config.agent_version
        assert "mlflow.linkedPrompts" in model.tags
        traces = client.search_traces(locations=[experiment.experiment_id])
        assert len(traces) == 6 and all(t.info.tags.get("mlflow.linkedPrompts") for t in traces)
        assert not (root / "logs/unit/tracing-error.log").exists()
        client.register_prompt("cad-builder", "Новая версия для будущего исследования")
        resolved = resolve_prompts(root, config)
        assert resolved.agent.prompt == config.agent.prompt  # новый номер не заменяет закреплённый
        client.delete_prompt_version("cad-builder", "1")
        model = Model()
        with pytest.raises(Exception, match="not found"):
            run((root, config), model, name="missing")
        assert model.calls == []
    with pytest.raises(ValidationError):
        PromptVersions(**{**config.prompts.model_dump(), "builder": "prompts:/cad-builder@latest"})


def test_versions_refuse_changed_content_before_request(sandbox) -> None:
    root, config = registered(sandbox)
    run((root, config), Model(), tracking=True)
    model = Model()
    changed = config.model_copy(update={"cad": config.cad.model_copy(update={"stl_tolerance_mm": 0.1})})
    with pytest.raises(ValueError, match="agent_version"):
        run((root, changed), model, name="changed-agent", tracking=True)
    assert not model.calls
    manifest = read_json(root / config.manifest)
    manifest["items"][0]["orientation"] += " Уточнение."
    write_json(root / config.manifest, manifest)
    with pytest.raises(ValueError, match="dataset_version"):
        run((root, config), model, name="changed-data", tracking=True)
    assert not model.calls


def test_publish_clone_prompts_files_traces_and_working_history(sandbox, monkeypatch) -> None:
    from mlflow.genai.datasets import search_datasets

    from mlflow import MlflowClient

    root, config = registered(sandbox)
    directory = run((root, config), Model(), tracking=True)
    before = {p.relative_to(directory): sha256(p) for p in directory.rglob("*") if p.is_file()}
    with FileLock(study_lock(root, "unit")), pytest.raises(Timeout):
        publish(root)
    snapshot = publish(root)
    assert "tracking.db" in snapshot["files"]
    assert all(sha256(directory / path) == value for path, value in before.items())
    clone = root.with_name(root.name + "-copy")
    shutil.copytree(root / "mlflow", clone / "mlflow", ignore=shutil.ignore_patterns(".runtime"))
    # Исходный путь физически исчезает: случайные абсолютные ссылки не смогут сработать.
    hidden = root.with_name(root.name + "-hidden")
    root.rename(hidden)
    try:
        monkeypatch.chdir(clone)
        verify_snapshot(clone)
        with local_tracking(clone):
            client = MlflowClient(tracking_uri=uri(clone), registry_uri=uri(clone))
            assert client.load_prompt(config.prompts.builder).template == config.agent.prompt
            experiment = client.get_experiment_by_name(config.name + " · " + config.title)
            assert len(client.search_runs([experiment.experiment_id])) == 6
            traces = client.search_traces(locations=[experiment.experiment_id])
            assert len(traces) == 6
            assert all(client.get_trace(t.info.trace_id).data.spans for t in traces)
            for dataset in search_datasets(experiment_ids=[experiment.experiment_id]):
                row = dataset.to_df().to_dict(orient="records")[0]
                for side, key in (("inputs", "png"), ("expectations", "stl")):
                    _, run_id, path = row[side][key].split("/", 2)
                    downloaded = Path(client.download_artifacts(run_id, path))
                    assert sha256(downloaded) == row[side][key + "_sha256"]
            # Независимая рабочая история не теряется при новом снимке.
            local_run = client.create_run(experiment.experiment_id, tags={"mlflow.runName": "local-only"})
        with sqlite3.connect(clone / "mlflow/tracking.db") as connection:
            connection.execute("update experiments set name='Published update' where experiment_id=0")
        snapshot["files"]["tracking.db"] = sha256(clone / "mlflow/tracking.db")
        write_json(clone / "mlflow/snapshot.json", snapshot)
        with pytest.raises(ValueError, match="Рабочая история сохранена"):
            runtime_database(clone)
        with sqlite3.connect(clone / "mlflow/.runtime/tracking.db") as connection:
            assert (
                connection.execute(
                    "select count(*) from runs where run_uuid=?", (local_run.info.run_id,)
                ).fetchone()[0]
                == 1
            )
        restore(clone)
        assert list((clone / ".local-archive").glob("mlflow-runtime-*/tracking.db"))
        # Изменение любого файла опубликованного набора обнаруживается.
        png = next((clone / "mlflow/files").rglob("DEV1.png"))
        png.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="изменён"):
            verify_snapshot(clone)
    finally:
        monkeypatch.chdir(ROOT)
        hidden.rename(root)


def test_debug_hashes_are_checked_and_cad_intermediates_removed(sandbox) -> None:
    from src.experiments.runner import build_recipe

    directory = run(sandbox, Model(), builder=build_recipe)
    trial = directory / "direct/DEV1-r01-a001"
    assert not list((trial / "debug").rglob("model.stl"))
    assert not list((trial / "debug").rglob("build-input.json"))
    record = read_record(trial)
    assert any(name.startswith("debug/") for name in record["files"])
    (trial / "debug/cad/cad.log").write_text("Подмена")
    with pytest.raises(ValueError, match="Артефакт изменён"):
        read_record(trial)
