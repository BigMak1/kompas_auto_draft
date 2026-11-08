"""Ошибки настоящего SDK на подставленном HTTP-ответе; сети и LLM в тестах нет."""

import asyncio
import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from PIL import Image

from src import utils
from src.agent import agent, cad
from src.config import load_config
from src.utils import Run, error_details, log_error, read_json, write_json


@pytest.mark.parametrize(
    "outcome", ["success", "incomplete", "failed", "server", "auth", "quota", "connection"]
)
def test_http_response_and_errors_are_saved_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, outcome: str
) -> None:
    secret = "sk-proj-synthetic-test-credential"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    calls = []
    payload = {
        "id": "resp_offline",
        "object": "response",
        "created_at": 0,
        "model": "offline",
        "status": outcome,
        "output": [],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "incomplete_details": {"reason": "max_output_tokens"} if outcome == "incomplete" else None,
        "error": {"code": "server_error", "message": "Backend unavailable"} if outcome == "failed" else None,
    }
    if outcome in {"auth", "quota", "server"}:
        payload = {
            "error": {
                "code": {"auth": "invalid_api_key", "quota": "insufficient_quota", "server": "server_error"}[
                    outcome
                ],
                "message": f"Synthetic API error; key={secret}",
                "type": "invalid_request_error",
                "param": None,
            }
        }
    if outcome == "success":
        recipe = {
            "steps": [
                {
                    "action": "add",
                    "z": 0,
                    "depth": 1,
                    "direction": "+Z",
                    "profile": {"kind": "circle", "x": 0, "y": 0, "radius": 5},
                }
            ]
        }
        payload.update(
            status="completed",
            output=[
                {
                    "id": "message_offline",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps(recipe), "annotations": []}],
                }
            ],
        )

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert len(calls) == 1, "Повторный запрос запрещён"
        assert request.url == "https://api.openai.com/v1/responses"
        body = json.loads(request.content)
        content = body["input"][0]["content"]
        assert len(content) == 1 and content[0]["type"] == "input_image"
        assert content[0]["detail"] == "high"
        prefix, encoded = content[0]["image_url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        sent_image = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(sent_image)) as image:
            image.load()
            assert image.format == "PNG" and image.size == (8, 4)
        saved_input = read_json(run.artifacts / "run.json")["input"]
        assert saved_input["sent_sha256"] == hashlib.sha256(sent_image).hexdigest()
        if outcome == "connection":
            raise httpx.ConnectError("Connection refused in offline test", request=request)
        status = {"auth": 401, "quota": 429, "server": 500}.get(outcome, 200)
        return httpx.Response(status, json=payload, headers={"x-request-id": "req_offline"})

    def http_client(**kwargs: object) -> httpx.AsyncClient:
        return DefaultAsyncHttpxClient(transport=httpx.MockTransport(respond), **kwargs)

    def build(recipe: object, directory: Path, settings: object) -> dict:
        assert outcome == "success", "При ошибке ответа построение не начинается"
        (directory / "model.stl").write_bytes(b"offline model")
        return {"solid_valid": True}

    # Настоящий клиент может обращаться только к подставленному транспорту.
    monkeypatch.setattr(agent, "AsyncOpenAI", AsyncOpenAI)
    monkeypatch.setattr(agent, "DefaultAsyncHttpxClient", http_client)
    monkeypatch.setattr(cad, "build", build)
    source = tmp_path / "drawing.png"
    Image.new("RGB", (20, 10), "white").save(source)
    run = Run("OFFLINE", root=tmp_path)
    config = load_config()
    config.agent.image_max_side = 8
    record = asyncio.run(agent.generate(source, run, config))
    assert len(calls) == 1
    if outcome == "success":
        assert record["status"] == "success" and record["cad_builds"] == 1
        assert read_json(run.artifacts / "response.json")["status"] == "completed"
        assert record["responses"][0]["response_id"] == "resp_offline"
        assert record["usage"]["total_tokens"] == 30
        assert not (run.artifacts / "error.log").exists()
        return
    assert record["cad_builds"] == 0
    assert record["status"] == "failed" and record["stage"] == "llm"
    assert record["failure_kind"] == ("generation" if outcome == "incomplete" else "infrastructure")
    assert not (run.directory / "model.stl").exists()
    saved = read_json(run.artifacts / "run.json")
    assert saved == record
    expected = {
        "incomplete": "max_output_tokens",
        "failed": "server_error",
        "server": "server_error",
        "auth": "invalid_api_key",
        "quota": "insufficient_quota",
        "connection": "Connection refused in offline test",
    }[outcome]
    for path in (run.logfile, run.artifacts / "error.log"):
        assert expected in path.read_text()
    if outcome != "connection":
        assert record["error"]["request_id"] == "req_offline"
        assert (run.artifacts / "response.json").is_file()
    if outcome in {"incomplete", "failed"}:
        assert record["usage"]["output_tokens"] == 20
        assert record["responses"] == []  # SDK отклонил ответ до on_llm_end.
    if outcome == "connection":
        assert record["error"]["causes"][-1]["type"] == "ConnectError"
    terminal = capsys.readouterr()
    logs = (
        terminal.out
        + terminal.err
        + "".join(
            p.read_text()
            for folder in (run.directory, run.artifacts)
            for p in folder.iterdir()
            if p.is_file()
        )
    )
    assert secret not in logs and "data:image/png;base64," not in logs


def test_recipe_validation_names_the_field(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    write_json(
        recipe,
        {
            "steps": [
                {
                    "action": "add",
                    "z": 0,
                    "depth": -1,
                    "direction": "+Z",
                    "profile": {"kind": "circle", "x": 0, "y": 0, "radius": 5},
                }
            ]
        },
    )
    run = Run("OFFLINE", root=tmp_path)
    record = agent.replay(recipe, run, load_config())
    assert record["stage"] == "recipe" and record["cad_builds"] == 0
    assert "steps.0.depth" in run.logfile.read_text()
    assert record["error"]["validation"][0]["loc"] == ["steps", 0, "depth"]


def test_recipe_save_error_is_not_a_generation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = tmp_path / "recipe.json"
    write_json(
        recipe,
        {
            "steps": [
                {
                    "action": "add",
                    "z": 0,
                    "depth": 1,
                    "direction": "+Z",
                    "profile": {"kind": "circle", "x": 0, "y": 0, "radius": 5},
                }
            ]
        },
    )
    run = Run("OFFLINE", root=tmp_path)

    def denied(path: Path, value: object) -> None:
        if path == run.directory / "recipe.json":
            raise PermissionError("Нельзя записать рецепт")
        write_json(path, value)

    monkeypatch.setattr(agent, "write_json", denied)
    record = agent.replay(recipe, run, load_config())
    assert record["stage"] == "recipe_save" and record["failure_kind"] == "infrastructure"
    assert record["cad_builds"] == 0 and record["status"] == "failed"
    assert "Нельзя записать рецепт" in run.logfile.read_text()


def test_diagnostics_redact_secrets_including_exception_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    secret = "synthetic-without-standard-prefix"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    try:
        try:
            raise ValueError(f"key={secret}; Authorization: Bearer abc.def; data:image/png;base64,YWJjZA==")
        except ValueError as cause:
            raise RuntimeError("bad key sk-proj-synthetic-other") from cause
    except RuntimeError as error:
        details = error_details(error, "получение ответа LLM")
    logfile, diagnostic = tmp_path / "run.log", tmp_path / "error.log"
    log_error(details, diagnostic, logfile)
    text = json.dumps(details, ensure_ascii=False) + logfile.read_text() + diagnostic.read_text()
    text += capsys.readouterr().out
    for forbidden in (secret, "sk-proj-synthetic-other", "abc.def", "YWJjZA=="):
        assert forbidden not in text
    assert "[СКРЫТО]" in text and "RuntimeError" in text and "ValueError" in text


def test_diagnostics_write_failure_does_not_hide_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def denied(*args: object) -> None:
        raise PermissionError("Нет доступа к файлу")

    monkeypatch.setattr(utils, "atomic", denied)
    log_error(error_details(ValueError("Первичная ошибка"), "проверка рецепта"), tmp_path / "error.log")
    output = capsys.readouterr().out
    assert "Первичная ошибка" in output and "Не удалось сохранить диагностику" in output
