"""Офлайн-проверки SDK и построителя без обращения к внешним сервисам."""

import asyncio
import json
import math
from pathlib import Path

import pytest
import trimesh
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from PIL import Image
from pydantic import ValidationError

from src.agent import agent, cad
from src.agent.schema import Recipe
from src.config import Config, load_config
from src.utils import Run, read_json, sha256, write_json

RECIPE = {
    "steps": [
        {
            "action": "add",
            "z": 0,
            "depth": 3,
            "direction": "+Z",
            "profile": {"kind": "circle", "x": 0, "y": 0, "radius": 5},
        }
    ]
}


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Настоящий OpenAI API запрещён в тестах")

    monkeypatch.setattr(agent, "AsyncOpenAI", forbidden)


class OfflineModel(Model):
    def __init__(self, text: str = json.dumps(RECIPE), error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls = 0

    async def get_response(self, **kwargs: object) -> ModelResponse:
        self.calls += 1
        assert self.calls == 1, "Повторный запрос запрещён"
        assert not kwargs["tools"]
        content = kwargs["input"][0]["content"]
        assert len(content) == 1 and content[0]["type"] == "input_image"
        assert content[0]["image_url"].startswith("data:image/png;base64,")
        if self.error:
            raise self.error
        return ModelResponse(
            output=[
                ResponseOutputMessage(
                    id="offline",
                    role="assistant",
                    type="message",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text=self.text, annotations=[])],
                )
            ],
            usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            response_id="offline",
        )

    async def stream_response(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Потоковый режим не используется")


def install_model(monkeypatch: pytest.MonkeyPatch, model: OfflineModel, config: Config) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["max_retries"] == 0
            assert kwargs["timeout"] == config.agent.timeout_seconds
            assert kwargs["base_url"] == "https://api.openai.com/v1"
            self.http_client = kwargs["http_client"]

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *args: object) -> None:
            await self.http_client.aclose()

    def create_model(**kwargs: object) -> OfflineModel:
        assert kwargs["model"] == config.agent.model
        return model

    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setattr(agent, "AsyncOpenAI", Client)
    monkeypatch.setattr(agent, "OpenAIResponsesModel", create_model)


def test_config_and_agent_surface() -> None:
    settings = load_config().agent.model_copy(
        update={
            "model": "offline-model",
            "reasoning_effort": "low",
            "prompt": "Проверка",
            "max_output_tokens": 128,
        }
    )
    configured = agent.make_agent(settings)
    assert configured.tools == [] and not configured.handoffs and not configured.mcp_servers
    assert configured.output_type is Recipe and configured.model == "offline-model"
    assert configured.model_settings.reasoning.effort == "low" and configured.model_settings.max_tokens == 128
    assert configured.instructions == "Проверка"
    with pytest.raises(ValidationError):
        Recipe.model_validate(RECIPE | {"path": "data/target/KF0011.stl"})


@pytest.mark.parametrize("outcome", ["success", "invalid_json", "cad_failure", "api_failure"])
def test_one_response_and_separate_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, outcome: str
) -> None:
    config = load_config()
    model = OfflineModel(
        "{" if outcome == "invalid_json" else json.dumps(RECIPE),
        ConnectionError("Соединение прервано") if outcome == "api_failure" else None,
    )
    install_model(monkeypatch, model, config)
    builds = []

    def build(recipe: Recipe, directory: Path, settings: object) -> dict:
        builds.append(recipe)
        assert settings == config.cad
        if outcome == "cad_failure":
            raise ValueError("Невозможное построение")
        (directory / "model.stl").write_bytes(b"offline geometry placeholder")
        return {"solid_valid": True, "files": {"model.stl": sha256(directory / "model.stl")}}

    monkeypatch.setattr(cad, "build", build)
    source = tmp_path / "drawing.png"
    Image.new("RGB", (8, 8), "white").save(source)
    run = Run("KFtest", root=tmp_path)
    run.create()
    (run.directory / "model.stl").write_bytes(b"old model")
    write_json(run.artifacts / "evaluation.json", {"iou": 1, "f_score": 1})
    run.report.parent.mkdir(parents=True)
    run.report.write_text("old scores")
    record = asyncio.run(agent.generate(source, run, config))
    assert model.calls == 1
    assert len(builds) == (1 if outcome in ("success", "cad_failure") else 0)
    assert record["cad_builds"] == len(builds)
    assert record["status"] == ("success" if outcome == "success" else "failed")
    assert read_json(run.artifacts / "run.json") == record
    assert not (run.directory / "run.json").exists()
    assert not (run.directory / "model.step").exists()
    assert not run.report.exists() and not (run.artifacts / "evaluation.json").exists()
    if outcome != "success":
        assert not (run.directory / "model.stl").exists()
    if outcome != "success":
        details = record["error"]
        assert details["type"] == record["error_type"]
        assert details["message"] in (run.artifacts / "error.log").read_text()
        assert "Traceback (most recent call last)" in (run.artifacts / "error.log").read_text()
        assert f"Этап: {details['stage']}" in run.logfile.read_text()
    if outcome == "api_failure":
        assert "Соединение прервано" in run.logfile.read_text()
        assert record["failure_kind"] == "infrastructure"
    if outcome == "cad_failure":
        assert record["stage"] == "cad" and "Невозможное построение" in run.logfile.read_text()
    terminal = capsys.readouterr().out
    assert "ПРОГОН" in terminal and '"responses"' not in terminal
    assert "\x1b" not in run.logfile.read_text()
    if outcome == "success":
        assert sorted(p.name for p in run.directory.iterdir()) == ["model.stl", "recipe.json", "run.log"]
    if outcome == "invalid_json":
        assert record["responses"][0]["output"][0]["content"][0]["text"] == "{"
        assert record["stage"] == "recipe"
        assert record["error"]["causes"][-1]["validation"][0]["type"] == "json_invalid"
        assert "EOF" in run.logfile.read_text()


def test_missing_key_preserves_previous_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run = Run("KFtest", root=tmp_path)
    run.create()
    (run.directory / "model.stl").write_bytes(b"previous")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        asyncio.run(agent.generate(tmp_path / "absent.png", run, load_config()))
    assert (run.directory / "model.stl").read_bytes() == b"previous"


def test_overwrite_does_not_touch_another_part(tmp_path: Path) -> None:
    first, second = Run("A", root=tmp_path), Run("B", root=tmp_path)
    for run in (first, second):
        run.create()
        (run.directory / "model.stl").write_bytes(run.pair_id.encode())
        write_json(run.artifacts / "run.json", {"part": run.pair_id})
    first.create()
    assert not (first.directory / "model.stl").exists()
    assert (second.directory / "model.stl").read_bytes() == b"B"
    assert read_json(second.artifacts / "run.json") == {"part": "B"}


def test_replay_can_overwrite_its_own_recipe(tmp_path: Path) -> None:
    run = Run("KFtest", root=tmp_path)
    run.create()
    source = run.directory / "recipe.json"
    write_json(source, RECIPE)
    (run.directory / "model.stl").write_bytes(b"old model")
    run.logfile.write_text("old log\n")
    write_json(run.artifacts / "evaluation.json", {"iou": 1, "f_score": 1})
    run.report.parent.mkdir(parents=True)
    run.report.write_text("old scores")
    record = agent.replay(source, run, load_config())
    assert record["status"] == "success"
    assert read_json(source) == RECIPE
    assert trimesh.load_mesh(run.directory / "model.stl").is_volume
    assert "old log" not in run.logfile.read_text()
    assert not (run.artifacts / "evaluation.json").exists() and not run.report.exists()
    assert sorted(p.name for p in run.directory.iterdir()) == ["model.stl", "recipe.json", "run.log"]


@pytest.mark.parametrize("interrupt", [False, True])
def test_deadline_and_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool) -> None:
    class SlowModel(OfflineModel):
        async def get_response(self, **kwargs: object) -> ModelResponse:
            self.calls += 1
            if interrupt:
                raise asyncio.CancelledError
            await asyncio.Event().wait()
            raise AssertionError("Ожидание должно прерваться")

    config = load_config()
    config.agent.timeout_seconds = 0.01
    model = SlowModel()
    install_model(monkeypatch, model, config)
    source = tmp_path / "drawing.png"
    Image.new("RGB", (8, 8), "white").save(source)
    run = Run("KFtest", root=tmp_path)
    if interrupt:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(agent.generate(source, run, config))
    else:
        asyncio.run(agent.generate(source, run, config))
    record = read_json(run.artifacts / "run.json")
    assert record["status"] == ("interrupted" if interrupt else "failed")
    assert record["cad_builds"] == 0 and model.calls == 1
    assert record["stage"] == "llm"
    assert record["error"]["type"] == ("CancelledError" if interrupt else "TimeoutError")
    assert (run.artifacts / "error.log").is_file()


@pytest.mark.parametrize(
    "change",
    [
        {"action": "cut"},
        {"z": -1},
        {"depth": 0},
        {"depth": float("nan")},
        {"profile": {"kind": "polygon", "points": [{"x": 0, "y": 0}] * 3}},
    ],
)
def test_invalid_recipe(change: dict) -> None:
    with pytest.raises(ValidationError):
        Recipe.model_validate({"steps": [RECIPE["steps"][0] | change]})


def test_real_cad_outputs_only_stl(tmp_path: Path) -> None:
    recipe = Recipe.model_validate(
        {
            "steps": [
                RECIPE["steps"][0],
                RECIPE["steps"][0]
                | {"action": "cut", "profile": {"kind": "circle", "x": 0, "y": 0, "radius": 2}},
            ]
        }
    )
    result = cad.build(recipe, tmp_path, load_config().cad)
    assert result["volume_mm3"] == pytest.approx(math.pi * (25 - 4) * 3, rel=1e-8)
    assert trimesh.load_mesh(tmp_path / "model.stl").is_volume
    assert [p.name for p in tmp_path.iterdir()] == ["model.stl"]


@pytest.mark.parametrize("action", ["add", "cut"])
def test_disconnected_and_noop_rejected(action: str) -> None:
    with pytest.raises(ValueError):
        cad.build_shape(
            Recipe.model_validate(
                {
                    "steps": [
                        RECIPE["steps"][0],
                        RECIPE["steps"][0]
                        | {
                            "action": action,
                            "profile": {"kind": "circle", "x": 20, "y": 0, "radius": 2},
                        },
                    ]
                }
            )
        )
