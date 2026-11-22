"""Контроль исследования без OpenAI: изоляция, бюджеты, история и знаменатель."""

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
import yaml
from PIL import Image
from pydantic import ValidationError

from src.agent.schemas.passport import Passport
from src.agent.schemas.recipe import Recipe
from src.agent.stages import messages
from src.config import ROOT
from src.experiments.config import ExperimentConfig, check_manifest, load_experiment
from src.experiments.report import aggregate, evaluate_study, observations, physical_usage, report_study
from src.experiments.runner import build_recipe, run_matrix
from src.experiments.store import read_record, verify_study
from src.utils import error_details, read_json, sha256, write_json

RECIPE = {
    "steps": [
        {
            "action": "add",
            "profile": {"kind": "rectangle", "x": -1, "y": -1, "width": 2, "height": 2},
            "z": 0,
            "depth": 2,
            "direction": "+Z",
        }
    ]
}
PASSPORT = {
    "schema_version": "drawing-passport-2",
    "views": [{"id": "V1", "label": "Главный вид", "kind": "view", "description": "Прямоугольник"}],
    "coordinates": {"origin": "Центр", "axis_z": "Толщина", "front_view_id": "V1", "status": "observed"},
    "features": [
        {
            "id": "F1",
            "kind": "брусок",
            "description": "Основное тело",
            "view_ids": ["V1"],
            "status": "observed",
        }
    ],
    "dimensions": [
        {
            "id": "D1",
            "name": "Толщина",
            "raw": "2",
            "value": 2,
            "unit": "mm",
            "view_id": "V1",
            "feature_id": "F1",
            "reading_status": "observed",
            "binding_status": "observed",
        }
    ],
    "relations": [],
    "unresolved": [],
    "general_notes": [],
}


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, ExperimentConfig]:
    config = load_experiment().model_copy(update={"validation_size": 1, "prompts": None})
    config = config.model_copy(
        update={
            "agent": config.agent.model_copy(
                update={"prompt": "Построй рецепт", "prompt_version": "fixture-builder"}
            ),
            "reading": config.reading.model_copy(
                update={
                    "prompt": "Прочитай чертёж",
                    "prompt_version": "fixture-reader",
                    "passport_format": "Паспорт",
                    "notes_format": "Заметки",
                }
            ),
        }
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src/code.py").write_text("# frozen code\n")
    for name in ("evaluate.py", "config.py", "utils.py"):
        (tmp_path / "src" / name).write_bytes((ROOT / "src" / name).read_bytes())
    for name in ("pyproject.toml", "uv.lock"):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    items = []
    for pair, split in (("DEV1", "dev"), ("VAL1", "val")):
        image = tmp_path / "data/input" / f"{pair}.png"
        mesh_path = tmp_path / "data/target" / f"{pair}.stl"
        image.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16 if split == "dev" else 17, 16), "white").save(image)
        mesh = trimesh.creation.box(extents=(2 if split == "dev" else 3, 2, 2))
        mesh.apply_translation((0, 0, 1))
        mesh.export(mesh_path)
        items.append(
            {
                "pair_id": pair,
                "split": split,
                "family": pair,
                "sheet_count": 1,
                "orientation": f"Ось Z, база — торец с пазом; паз к +Y. Подсказка {pair}.",
                "review": {"status": "accepted"},
                "input_png": {"path": f"input/{pair}.png", "sha256": sha256(image)},
                "target_mesh": {"path": f"target/{pair}.stl", "sha256": sha256(mesh_path)},
            }
        )
    write_json(
        tmp_path / config.manifest, {"status": "ready", "coordinates": {"units": "mm"}, "items": items}
    )
    config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"surface_samples": 32})}
    )
    return tmp_path, config


class Model:
    def __init__(self) -> None:
        self.calls = []
        self.fail_type = None
        self.reading_seconds = 0.01

    async def __call__(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        schema = kwargs["output_type"]
        usage = {"input_tokens": 100, "output_tokens": 900 if schema is Recipe else 700}
        if self.fail_type:
            return {
                "status": "failed",
                "failure_kind": self.fail_type,
                "seconds": 0.01,
                "usage": usage,
                "error": error_details(ValueError("fixture failure"), "LLM"),
            }
        output = copy.deepcopy(
            PASSPORT if schema is Passport else RECIPE if schema is Recipe else "Толщина 2 мм"
        )
        return {
            "status": "success",
            "usage": usage,
            "seconds": self.reading_seconds if schema is not Recipe else 0.01,
            "output": output,
            "http_responses": 1,
        }


async def cube_builder(
    recipe: dict, directory: Path, technical: Path, config: ExperimentConfig, timeout: float
) -> dict:
    assert timeout > 0
    Recipe.model_validate(recipe)
    mesh = trimesh.creation.box(extents=(2, 2, 2))
    mesh.apply_translation((0, 0, 1))
    mesh.export(directory / "model.stl")
    return {
        "status": "success",
        "cad": {"files": {"model.stl": sha256(directory / "model.stl")}, "solid_valid": True},
    }


def run(
    sandbox: tuple[Path, ExperimentConfig],
    model: Model,
    *,
    name: str = "unit",
    split: str = "dev",
    repeats: int = 1,
    retry: bool = False,
    builder: Any = cube_builder,
    tracking: bool = False,
) -> Path:
    root, config = sandbox
    return asyncio.run(
        run_matrix(
            root,
            config,
            name,
            split,
            None,
            repeats,
            retry=retry,
            requester=model,
            builder=builder,
            tracking=tracking,
        )
    )


def test_passport_unknown_and_references() -> None:
    Passport.model_validate(PASSPORT)
    for change in (
        lambda p: p["dimensions"][0].update(reading_status="unresolved"),
        lambda p: p["dimensions"][0].update(view_id="absent"),
        lambda p: p["dimensions"][0].update(id="F1"),
        lambda p: p["dimensions"][0].update(binding_status="unresolved"),
        lambda p: p["dimensions"][0].update(value=float("nan")),
    ):
        value = copy.deepcopy(PASSPORT)
        change(value)
        with pytest.raises(ValidationError):
            Passport.model_validate(value)


def test_only_passport_context_contains_no_image() -> None:
    context = messages(None, json.dumps(PASSPORT), "high")
    assert [item["type"] for item in context[0]["content"]] == ["input_text"]
    assert "image_url" not in json.dumps(context)


def test_four_routes_shared_passport_budgets_and_repeats(sandbox: tuple[Path, ExperimentConfig]) -> None:
    model = Model()
    directory = run(sandbox, model, repeats=3)
    assert len(model.calls) == 18  # 3 × (direct + passport + 2 builders + notes + builder)
    assert sum(c["output_type"] is Passport for c in model.calls) == 3
    assert sum(c["output_type"] is None for c in model.calls) == 3
    builders = [c for c in model.calls if c["output_type"] is Recipe]
    assert sum(c["image"] is None for c in builders) == 3
    assert all(c["text"] is not None for c in builders if c["image"] is None)
    orientation = read_json(sandbox[0] / sandbox[1].manifest)["items"][0]["orientation"]
    assert all(c["text"].split("\n\n", 1)[0] == orientation for c in model.calls)
    assert all(c["text"] == orientation for c in model.calls if c["output_type"] is not Recipe)
    assert sum(c["text"] == orientation for c in builders) == 3
    assert all(c["max_tokens"] == (16384 if c["text"] == orientation else 15684) for c in builders)
    for repeat in range(1, 4):
        a = read_record(directory / "passport_only" / f"DEV1-r{repeat:02d}-a001")
        b = read_record(directory / "passport_png" / f"DEV1-r{repeat:02d}-a001")
        assert a["reading_path"] == b["reading_path"]
        assert a["reading_sha256"] == b["reading_sha256"]
        assert a["usage"]["output_tokens"] == 1600
        for variant in ("direct", "passport_only", "passport_png", "notes_png"):
            assert read_record(directory / variant / f"DEV1-r{repeat:02d}-a001")["orientation"] == orientation
        assert read_record(directory / a["reading_path"])["orientation"] == orientation
        assert json.loads((directory / a["reading_path"] / "passport.json").read_text()) == PASSPORT
    assert physical_usage(directory)["output_tokens"] == 3 * (4 * 900 + 2 * 700)
    run(sandbox, model, repeats=3)
    assert len(model.calls) == 18


def test_orientation_does_not_forward_target_metadata(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, config = sandbox
    manifest = read_json(root / config.manifest)
    item = manifest["items"][0]
    hidden = "TARGET-ONLY-DATA-DO-NOT-SEND"
    item["limitations"] = [hidden]
    item["review"]["evidence"] = hidden
    item["preparation"] = {"transform_to_target": np.eye(4).tolist(), "source_mesh": hidden}
    write_json(root / config.manifest, manifest)
    model = Model()
    run(sandbox, model)
    for call in model.calls:
        context = messages(call["image"], call["text"], "high")
        serialized = json.dumps(context, ensure_ascii=False)
        assert hidden not in serialized and item["target_mesh"]["sha256"] not in serialized
        assert "target_mesh" not in serialized and "preparation" not in serialized
        if call["image"] is None:
            assert call["output_type"] is Recipe
            assert "input_image" not in serialized and "image_url" not in serialized
            assert json.loads(call["text"].split("\n\n", 1)[1]) == PASSPORT


@pytest.mark.parametrize("orientation", [None, "", "  ", {"target": "secret"}])
def test_missing_orientation_stops_before_requests(
    sandbox: tuple[Path, ExperimentConfig], orientation: object
) -> None:
    root, config = sandbox
    manifest = read_json(root / config.manifest)
    manifest["items"][0]["orientation"] = orientation
    write_json(root / config.manifest, manifest)
    model = Model()
    with pytest.raises(ValueError, match="ориентация"):
        run(sandbox, model)
    assert not model.calls and not (root / "run").exists()


def test_changed_orientation_requires_new_study(sandbox: tuple[Path, ExperimentConfig]) -> None:
    model = Model()
    directory = run(sandbox, model)
    before = sha256(directory / "study.json")
    root, config = sandbox
    manifest = read_json(root / config.manifest)
    manifest["items"][0]["orientation"] = "Другой торец на Z=0"
    write_json(root / config.manifest, manifest)
    with pytest.raises(ValueError, match="изменились"):
        run(sandbox, model)
    assert len(model.calls) == 6 and sha256(directory / "study.json") == before


@pytest.mark.parametrize("scale", [-1, 1.1])
def test_preparation_rejects_reflection_and_scaling(
    sandbox: tuple[Path, ExperimentConfig], scale: float
) -> None:
    root, config = sandbox
    manifest = read_json(root / config.manifest)
    matrix = np.eye(4)
    matrix[0, 0] = scale
    manifest["items"][0]["preparation"] = {"transform_to_target": matrix.tolist()}
    write_json(root / config.manifest, manifest)
    model = Model()
    with pytest.raises(ValueError, match="только поворот и перенос"):
        run(sandbox, model)
    assert not model.calls


def test_changed_config_or_code_cannot_resume(sandbox: tuple[Path, ExperimentConfig]) -> None:
    model = Model()
    directory = run(sandbox, model)
    (sandbox[0] / "src/code.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="изменились"):
        run(sandbox, model)
    assert len(model.calls) == 6
    assert (directory / "source.zip").is_file()


def test_failure_stays_zero_and_cannot_be_retried_as_infrastructure(
    sandbox: tuple[Path, ExperimentConfig],
) -> None:
    model = Model()
    model.fail_type = "generation"
    directory = run(sandbox, model)
    first_calls = len(model.calls)
    run(sandbox, model, retry=True)
    assert len(model.calls) == first_calls
    evaluate_study(sandbox[0], "unit")
    study = verify_study(directory)
    rows = observations(sandbox[0], directory, study)
    result = aggregate(rows, study)
    assert result["complete"]
    assert all(v["iou"] == 0 and v["f_score"] == 0 for v in result["variants"].values())


def test_infrastructure_missing_and_explicit_retry_preserves_first(
    sandbox: tuple[Path, ExperimentConfig],
) -> None:
    model = Model()
    model.fail_type = "infrastructure"
    directory = run(sandbox, model)
    originals = {p: p.read_bytes() for p in directory.glob("*/*/record.json")}
    evaluate_study(sandbox[0], "unit")
    study = verify_study(directory)
    assert not aggregate(observations(sandbox[0], directory, study), study)["complete"]
    model.fail_type = None
    run(sandbox, model, retry=True)
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert len(list((directory / "direct").glob("*-a*"))) == 2
    evaluate_study(sandbox[0], "unit")
    assert aggregate(observations(sandbox[0], directory, study), study)["complete"]


def test_expired_reading_budget_stops_recipe_and_cad(sandbox: tuple[Path, ExperimentConfig]) -> None:
    model = Model()
    model.reading_seconds = 901
    directory = run(sandbox, model)
    assert len(model.calls) == 3  # direct, one passport, one notes
    for variant in ("passport_only", "passport_png", "notes_png"):
        record = read_record(directory / variant / "DEV1-r01-a001")
        assert record["failure_kind"] == "generation" and record["cad_builds"] == 0


def test_missing_cell_and_tampered_artifact_never_shrink_denominator(
    sandbox: tuple[Path, ExperimentConfig],
) -> None:
    directory = run(sandbox, Model())
    evaluate_study(sandbox[0], "unit")
    study = verify_study(directory)
    original = observations(sandbox[0], directory, study)
    assert aggregate(original, study)["complete"]
    model = directory / "direct/DEV1-r01-a001/model.stl"
    model.write_bytes(model.read_bytes() + b"changed")
    rows = observations(sandbox[0], directory, study)
    summary = aggregate(rows, study)
    assert summary["expected"] == 4 and summary["measured"] == 3
    assert summary["variants"]["direct"]["iou"] is None


def test_dataset_hashes_and_family_overlap(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, config = sandbox
    check_manifest(root, config)
    manifest = read_json(root / config.manifest)
    manifest["items"][1]["family"] = manifest["items"][0]["family"]
    write_json(root / config.manifest, manifest)
    with pytest.raises(ValueError, match="Родственные"):
        check_manifest(root, config)
    manifest["items"][1]["family"] = "val"
    write_json(root / config.manifest, manifest)
    (root / "data/input/VAL1.png").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="изменён"):
        check_manifest(root, config)


@pytest.mark.parametrize("sheet_count", [None, 2, True])
def test_unconfirmed_or_multiple_sheets_stop_before_requests(
    sandbox: tuple[Path, ExperimentConfig], sheet_count: object
) -> None:
    root, config = sandbox
    manifest = read_json(root / config.manifest)
    manifest["items"][0]["sheet_count"] = sheet_count
    write_json(root / config.manifest, manifest)
    model = Model()
    with pytest.raises(ValueError, match="ровно один лист"):
        run(sandbox, model)
    assert model.calls == [] and not (root / "run").exists()


def test_offline_report_and_repeated_evaluation_match(sandbox: tuple[Path, ExperimentConfig]) -> None:
    directory = run(sandbox, Model())
    evaluate_study(sandbox[0], "unit")
    first = observations(sandbox[0], directory, verify_study(directory))
    evaluate_study(sandbox[0], "unit")
    second = observations(sandbox[0], directory, verify_study(directory))
    assert [(r["iou"], r["f_score"]) for r in first] == [(r["iou"], r["f_score"]) for r in second]
    result = report_study(sandbox[0], "unit")
    assert result["complete"] and not result["formal_validation"]
    assert (sandbox[0] / "results/unit/runs.csv").is_file()
    assert (sandbox[0] / "results/unit/metrics.png").is_file()


def test_cad_worker_builds_and_obeys_timeout(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, config = sandbox
    destination = root / "worker-output"
    destination.mkdir()
    result = asyncio.run(build_recipe(RECIPE, destination, root / "worker", config, 30))
    assert result["status"] == "success"
    assert trimesh.load_mesh(destination / "model.stl").is_volume
    with pytest.raises(TimeoutError):
        asyncio.run(build_recipe(RECIPE, destination, root / "worker-timeout", config, 0.001))


@pytest.mark.parametrize(
    ("outcome", "tracing"),
    [
        (name, False)
        for name in (
            "passport",
            "invalid_link",
            "invalid_unit",
            "invalid_json",
            "server",
            "incomplete",
            "recipe",
            "notes",
        )
    ]
    + [("notes", True), ("incomplete", True)],
)
def test_sdk_stage_is_one_request_and_keeps_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    tracing: bool,
    sandbox: tuple[Path, ExperimentConfig],
) -> None:
    from contextlib import nullcontext

    import httpx2 as httpx
    from openai import DefaultAsyncHttpxClient

    from src.agent import stages

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-offline-experiment-test")
    output_type = Recipe if outcome == "recipe" else None if outcome == "notes" else Passport
    content = copy.deepcopy(RECIPE if output_type is Recipe else PASSPORT)
    if outcome == "invalid_link":
        content["dimensions"][0]["feature_id"] = "absent"
    if outcome == "invalid_unit":
        content["dimensions"][0]["unit"] = None
    payload = {
        "id": "resp_offline",
        "object": "response",
        "created_at": 0,
        "model": "offline",
        "status": "completed",
        "tools": [],
        "output": [
            {
                "id": "message_offline",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Размер не прочитан" if output_type is None else json.dumps(content),
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "output_tokens_details": {"reasoning_tokens": 150},
        },
        "error": None,
        "incomplete_details": None,
    }
    if outcome == "invalid_json":
        payload["output"][0]["content"][0]["text"] = "{broken"
    if outcome == "incomplete":
        payload.update(status="incomplete", output=[], incomplete_details={"reason": "max_output_tokens"})
    if outcome == "server":
        payload = {
            "error": {
                "code": "server_error",
                "type": "server_error",
                "message": "offline error sk-proj-offline-experiment-test",
            }
        }
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            500 if outcome == "server" else 200, json=payload, headers={"x-request-id": "req_offline"}
        )

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return DefaultAsyncHttpxClient(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(stages, "DefaultAsyncHttpxClient", client)
    raw_directory = tmp_path
    context = nullcontext(stages.request)
    if tracing:
        from src.experiments.tracing import traced_requests

        directory = run(sandbox, Model())
        raw_directory = tmp_path / "run/unit/readings/DEV1-r01-notes-a001/debug"
        context = traced_requests(tmp_path, directory, stages.request)
    with context as requester:
        result = asyncio.run(
            requester(
                settings=load_experiment().agent,
                instructions="offline",
                image="data:image/png;base64,AAAA" if tracing else None,
                text=json.dumps(PASSPORT),
                output_type=output_type,
                max_tokens=8192,
                timeout=10,
                raw_directory=raw_directory,
            )
        )
    assert len(requests) == 1
    assert requests[0]["max_output_tokens"] == 8192
    assert requests[0]["tools"] == []
    if output_type is not None:
        assert requests[0]["text"]["format"]["type"] == "json_schema"
        assert requests[0]["text"]["format"]["strict"] is True
    assert requests[0]["input"][0]["content"][0]["type"] == "input_text"
    assert "previous_response_id" not in requests[0]
    assert result["status"] == ("success" if outcome in {"passport", "recipe", "notes"} else "failed")
    if outcome != "server":
        assert result["usage"]["output_tokens"] == 200  # reasoning уже входят, не прибавляем их повторно.
    expected_code = {
        "invalid_link": "invalid_references",
        "invalid_unit": "schema_validation",
        "invalid_json": "invalid_json",
        "server": "api_error",
    }
    if outcome in expected_code:
        assert result["error"]["code"] == expected_code[outcome]
    if outcome == "invalid_unit":
        assert result["error"]["validation"][0]["loc"] == ["dimensions", 0, "unit"]
    if outcome in {"invalid_link", "invalid_unit", "invalid_json", "server", "incomplete"}:
        assert result["error"]["request_id"] == "req_offline"
        assert result["failure_kind"] == ("infrastructure" if outcome == "server" else "generation")
    if outcome == "incomplete":
        assert result["failure_reason"] == "max_output_tokens"
        assert result["error"]["code"] == "max_output_tokens"
        assert "8192" in result["error"]["summary"] and "150 на рассуждения" in result["error"]["summary"]
    assert "sk-proj-offline-experiment-test" not in (raw_directory / "stage.json").read_text()
    assert (raw_directory / "response.json").is_file()
    if tracing:
        from mlflow.tracking import MlflowClient

        from src.experiments.tracking import uri

        trace = MlflowClient(tracking_uri=uri(tmp_path)).get_trace(result["trace_id"])
        assert any(s.name == "AgentRunner.run" for s in trace.data.spans)
        assert trace.info.tags["capture"] == "live_sdk"
        assert trace.info.trace_metadata["mlflow.sourceRun"]
        serialized = json.dumps(trace.to_dict(), ensure_ascii=False)
        assert "base64,AAAA" not in serialized and "sk-proj-offline-experiment-test" not in serialized
        assert any("[PNG СКРЫТ]" in str(span.inputs) for span in trace.data.spans)
        assert "mlflow-attachment://" not in serialized
        assert str(trace.info.state) == ("ERROR" if outcome == "incomplete" else "OK")


def test_tracing_keeps_shared_passport_and_links_every_request(
    sandbox: tuple[Path, ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlflow.tracking import MlflowClient

    import mlflow
    from src.experiments.tracking import uri

    monkeypatch.setenv("MLFLOW_TRACE_EXTRACT_ATTACHMENTS", "true")
    model = Model()
    directory = run(sandbox, model, tracking=True)
    assert len(model.calls) == 6
    assert sum(call["output_type"] is Passport for call in model.calls) == 1
    assert all(call["tracing"] for call in model.calls)
    client = MlflowClient(tracking_uri=uri(sandbox[0]))
    runs = client.search_runs(["1"])
    assert len(runs) == 6
    traces = client.search_traces(locations=["1"])
    assert len(traces) == 6
    assert {t.info.trace_metadata["mlflow.sourceRun"] for t in traces} == {r.info.run_id for r in runs}
    assert all(t.info.trace_metadata["mlflow.modelId"] for t in traces)
    assert len(list(directory.glob("readings/*passport*/record.json"))) == 1
    import os

    assert os.environ["MLFLOW_TRACE_EXTRACT_ATTACHMENTS"] == "true"
    assert mlflow.active_run() is None


def test_tracing_error_after_response_does_not_repeat_or_discard_it(
    sandbox: tuple[Path, ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow
    from src.experiments.tracing import traced_requests

    directory = run(sandbox, Model())
    model = Model()

    def unavailable(**kwargs: Any) -> None:
        raise OSError("fixture tracking outage")

    monkeypatch.setattr(mlflow, "update_current_trace", unavailable)
    with traced_requests(sandbox[0], directory, model) as requester:
        result = asyncio.run(
            requester(
                image=None,
                output_type=Recipe,
                max_tokens=16384,
                raw_directory=sandbox[0] / "run/unit/direct/DEV1-r01-a001/debug/llm",
            )
        )
    assert result["status"] == "success" and len(model.calls) == 1
    assert "fixture tracking outage" in (sandbox[0] / "logs/unit/tracing-error.log").read_text()


def test_mlflow_is_local_idempotent_and_restorable(
    sandbox: tuple[Path, ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlflow.tracking import MlflowClient

    from src.experiments.tracking import sync_study, uri

    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    directory = run(sandbox, Model())
    sync_study(directory, sandbox[0])
    client = MlflowClient(tracking_uri=uri(sandbox[0]))
    config = sandbox[1]
    experiment = client.get_experiment_by_name(config.name + " · " + config.title)
    runs = client.search_runs([experiment.experiment_id])
    times = {r.info.run_id: r.info.end_time for r in runs}
    sync_study(directory, sandbox[0])
    assert {r.info.run_id: r.info.end_time for r in client.search_runs([experiment.experiment_id])} == times
    assert len(runs) == 6
    assert all(r.inputs.dataset_inputs and client.get_run(r.info.run_id).inputs.model_inputs for r in runs)
    assert all(
        "agent.prompt_version" in r.data.params and "reading.prompt_version" in r.data.params for r in runs
    )
    assert len(client.search_logged_models([experiment.experiment_id])) == 1
    from mlflow.genai.datasets import search_datasets

    from src.experiments.tracking import local_tracking

    with local_tracking(sandbox[0]):
        datasets = search_datasets(experiment_ids=[experiment.experiment_id])
        assert len(datasets) == 2
        assert sum(len(d.to_df()) for d in datasets) == 2
    assert all(len(client.get_metric_history(r.info.run_id, "output_tokens")) == 1 for r in runs)
    evaluate_study(sandbox[0], "unit")
    sync_study(directory, sandbox[0])
    sync_study(directory, sandbox[0])
    runs = client.search_runs([experiment.experiment_id])
    assert sum("iou" in r.data.metrics for r in runs) == 4
    assert all(
        len(client.get_metric_history(r.info.run_id, "iou")) == 1
        for r in runs
        if r.data.tags["kind"] == "route"
    )


def test_evaluation_tampering_is_detected(sandbox: tuple[Path, ExperimentConfig]) -> None:
    directory = run(sandbox, Model())
    evaluate_study(sandbox[0], "unit")
    score = sandbox[0] / "results/unit/direct/DEV1-r01-a001/evaluation.json"
    value = read_json(score)
    value["iou"] = 0.123
    write_json(score, value)
    summary = aggregate(observations(sandbox[0], directory, verify_study(directory)), verify_study(directory))
    assert not summary["complete"] and summary["measured"] == 3


def test_missing_matrix_keeps_all_60_rows(sandbox: tuple[Path, ExperimentConfig]) -> None:
    from src.experiments.store import freeze

    root, config = sandbox
    manifest = read_json(root / config.manifest)
    manifest["items"] = [{**manifest["items"][1], "pair_id": f"VAL{i}"} for i in range(5)]
    directory, study = freeze(root, "empty", config, manifest, "val", None, 3)
    rows = observations(root, directory, study)
    assert len(rows) == 60
    summary = aggregate(rows, study)
    assert summary["expected"] == 60 and summary["measured"] == 0
    assert all(v["iou"] is None for v in summary["variants"].values())


def test_offline_evaluation_does_not_import_agents_cad_or_mlflow(
    sandbox: tuple[Path, ExperimentConfig],
) -> None:
    import subprocess
    import sys

    directory = run(sandbox, Model())
    command = (
        "import sys; from pathlib import Path; "
        'sys.modules.update({k:None for k in ["agents","openai","cadquery","mlflow"]}); '
        "from src.experiments.report import evaluate_study, report_study; "
        'evaluate_study(Path(sys.argv[1]),"unit"); report_study(Path(sys.argv[1]),"unit")'
    )
    completed = subprocess.run(
        [sys.executable, "-c", command, str(sandbox[0])], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert (directory / "source.zip").is_file()


def test_baseline_subset_matches_experiment_dev() -> None:
    baseline = read_json(ROOT / "data/manifest.json")["items"]
    current = check_manifest(ROOT, load_experiment())["items"]
    dev = {item["pair_id"]: item for item in current if item["split"] == "dev"}
    assert len(dev) == 10 and sum(item["split"] == "val" for item in current) == 5
    for item in baseline:
        assert item["split"] == "val"
        for field in (
            "input_png",
            "target_mesh",
            "original_cdw",
            "original_m3d",
            "limitations",
            "orientation",
            "preparation",
        ):
            assert dev[item["pair_id"]].get(field) == item.get(field)
    # Ранее использованные KF0025/KF0029 и пара с оставшимся замечанием не попадают в val.
    assert {"KF0025", "KF0029", "KF0012"} <= dev.keys()


def test_changed_evaluator_cannot_silently_recompute(sandbox: tuple[Path, ExperimentConfig]) -> None:
    from src.experiments.report import verify_evaluator

    directory = run(sandbox, Model())
    study = verify_study(directory)
    study["source_sha256"]["src/evaluate.py"] = "different"
    with pytest.raises(ValueError, match="Код оценщика"):
        verify_evaluator(study)


def test_cli_selected_config_controls_run_and_saved_snapshot(
    sandbox: tuple[Path, ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments import main, runner

    root, base = sandbox
    custom = base.model_copy(
        update={
            "name": "custom-study",
            "variants": ["direct"],
            "repeats": 2,
            "agent": base.agent.model_copy(update={"max_output_tokens": 12000}),
        }
    )
    path = root / "configs/experiments/custom.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(custom.model_dump(mode="json")), encoding="utf-8")
    model = Model()

    async def offline_run(*args: Any, **kwargs: Any) -> Path:
        return await run_matrix(*args, **kwargs, requester=model, builder=cube_builder, tracking=False)

    monkeypatch.setattr(main, "ROOT", root)
    monkeypatch.setattr(runner, "run_matrix", offline_run)
    assert main.main(["run", "--config", str(path), "--split", "dev", "--pair", "DEV1"]) == 0
    study = verify_study(root / "run/custom-study-dev-dev1-r2")
    assert study["config"] == custom.model_dump(mode="json")
    assert len(study["schedule"]) == len(model.calls) == 2
    assert all(call["max_tokens"] == 12000 for call in model.calls)
    transcript = (root / "logs/custom-study-dev-dev1-r2/run.log").read_text()
    assert "ПЛАН" in transcript and "CAD" in transcript and "ИТОГ СЕРИИ" in transcript


def test_catalogue_and_number_selection_without_api(capsys: pytest.CaptureFixture) -> None:
    from src.experiments import main
    from src.experiments.config import DEFAULT_CONFIG, resolve_experiment

    assert main.main(["list"]) == 0
    assert main.main(["show", "1"]) == 0
    text = capsys.readouterr().out
    assert "Влияние паспорта" in text and "Только паспорт" in text
    assert "notes_png" not in text and "hw2" not in text
    path, config = resolve_experiment("1", None)
    assert path == DEFAULT_CONFIG and config.number == 1
    with pytest.raises(ValueError, match="не найден"):
        resolve_experiment("99", None)


def test_numbered_run_reports_failures_and_keeps_original_attempts(
    sandbox: tuple[Path, ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments import main, runner

    root, config = sandbox
    model = Model()
    model.fail_type = "generation"

    async def offline_run(*args: Any, **kwargs: Any) -> Path:
        return await run_matrix(*args, **kwargs, requester=model, builder=cube_builder, tracking=False)

    monkeypatch.setattr(main, "ROOT", root)
    monkeypatch.setattr(main, "resolve_experiment", lambda *_: (root / "config.yaml", config))
    monkeypatch.setattr(runner, "run_matrix", offline_run)
    command = ["run", "1", "--split", "dev", "--pair", "dev1", "--repeats", "1", "--study", "debug"]
    assert main.main(command) == 1
    hashes = {p: sha256(p) for p in (root / "run/debug").rglob("*") if p.is_file()}
    count = len(model.calls)
    assert main.main(command) == 1
    assert len(model.calls) == count and all(sha256(p) == value for p, value in hashes.items())
    log_text = (root / "logs/debug/run.log").read_text()
    assert "Не удалось прочитать чертёж (общий паспорт)" in log_text
    assert "Новый запрос не отправляется" in log_text
    assert "Построено 0/4" in log_text


def test_package_help_without_generation_dependencies() -> None:
    import subprocess
    import sys

    command = (
        "import runpy, sys; "
        'sys.modules.update({k:None for k in ["agents","openai","cadquery","mlflow"]}); '
        'sys.argv=["src.experiments"]; runpy.run_module("src.experiments",run_name="__main__")'
    )
    completed = subprocess.run(
        [sys.executable, "-c", command], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    assert "run" in completed.stdout and "evaluate" in completed.stdout and "--help" in completed.stdout
