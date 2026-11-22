"""Офлайн-тесты единого LLM-клиента: Agents SDK + MockTransport, ресайз PNG и редакция секретов."""

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from PIL import Image

from src.core import logging as core_logging
from src.core.config import load_experiment
from src.core.logging import log_error
from src.core.utils import error_details
from src.llm import client as llm_client
from src.llm.image import prepare_image
from src.schemas.passport import Passport
from src.schemas.recipe import Recipe
from tests.conftest import PASSPORT, RECIPE


@pytest.mark.parametrize(
    ("size", "limit", "expected"),
    [
        ((80, 40), 32, (32, 16)),
        ((40, 80), 32, (16, 32)),
        ((16, 8), 32, (16, 8)),
        ((80, 40), None, (80, 40)),
    ],
)
def test_prepare_image_preserves_original_and_aspect_ratio(
    tmp_path: Path, size: tuple[int, int], limit: int | None, expected: tuple[int, int]
) -> None:
    source = tmp_path / "drawing.png"
    Image.new("RGB", size, "white").save(source)
    original = source.read_bytes()
    url, info = prepare_image(source, limit)
    decoded = base64.b64decode(url.split(",", 1)[1], validate=True)
    with Image.open(BytesIO(decoded)) as img:
        img.load()
        assert img.format == "PNG" and img.size == expected
    assert source.read_bytes() == original
    assert info["original_size"] == list(size) and info["sent_size"] == list(expected)
    assert info["resized"] == (size != expected)
    if size == expected:
        assert info["sent_sha256"] == info["sha256"]


@pytest.mark.parametrize(
    "outcome",
    [
        "success_recipe",
        "success_notes",
        "invalid_json",
        "invalid_unit",
        "incomplete",
        "server",
        "auth",
        "connection",
    ],
)
def test_llm_request_single_shot_diagnostics_and_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    secret = "sk-proj-synthetic-test-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    calls: list[httpx.Request] = []

    output_type = (
        None
        if outcome == "success_notes"
        else Passport
        if outcome in {"invalid_json", "invalid_unit"}
        else Recipe
    )

    if outcome in {"server", "auth"}:
        payload = {
            "error": {
                "code": "invalid_api_key" if outcome == "auth" else "server_error",
                "message": f"Synthetic error with key={secret}",
                "type": "invalid_request_error",
                "param": None,
            }
        }
    elif outcome == "incomplete":
        payload = {
            "id": "resp_inc",
            "object": "response",
            "created_at": 0,
            "model": "offline",
            "status": "incomplete",
            "output": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "output_tokens_details": {"reasoning_tokens": 50},
            },
            "incomplete_details": {"reason": "max_output_tokens"},
        }
    else:
        if outcome == "success_recipe":
            text = json.dumps(RECIPE)
        elif outcome == "success_notes":
            text = "Свободные заметки"
        elif outcome == "invalid_json":
            text = "{broken"
        else:
            bad = json.loads(json.dumps(PASSPORT))
            bad["dimensions"][0]["unit"] = None
            text = json.dumps(bad)
        payload = {
            "id": "resp_ok",
            "object": "response",
            "created_at": 0,
            "model": "offline",
            "status": "completed",
            "tools": [],
            "output": [
                {
                    "id": "msg_1",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140},
        }

    def respond(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert len(calls) == 1
        if outcome == "connection":
            raise httpx.ConnectError("Connection refused", request=req)
        status = 401 if outcome == "auth" else 500 if outcome == "server" else 200
        return httpx.Response(status, json=payload, headers={"x-request-id": "req_1"})

    monkeypatch.setattr(llm_client, "AsyncOpenAI", AsyncOpenAI)
    monkeypatch.setattr(
        llm_client,
        "DefaultAsyncHttpxClient",
        lambda **kw: DefaultAsyncHttpxClient(transport=httpx.MockTransport(respond), **kw),
    )

    png = tmp_path / "draw.png"
    Image.new("RGB", (8, 8), "white").save(png)
    img_url, _ = prepare_image(png, None)
    raw_dir = tmp_path / "debug"
    settings = load_experiment().agent

    res = asyncio.run(
        llm_client.request(
            settings=settings,
            instructions="Тест",
            image=img_url,
            text="Ориентация",
            output_type=output_type,
            max_tokens=1000,
            timeout=10.0,
            raw_directory=raw_dir,
        )
    )
    assert len(calls) == 1
    if outcome in {"success_recipe", "success_notes"}:
        assert res["status"] == "success"
    elif outcome in {"server", "auth", "connection"}:
        assert res["status"] == "failed" and res["failure_kind"] == "infrastructure"
    else:
        assert res["status"] == "failed" and res["failure_kind"] == "generation"
        if outcome == "incomplete":
            assert res["failure_reason"] == "max_output_tokens"
            assert res["error"]["code"] == "max_output_tokens"
        elif outcome == "invalid_json":
            assert res["error"]["code"] == "invalid_json"
        elif outcome == "invalid_unit":
            assert res["error"]["code"] == "schema_validation"
            assert "dimensions.0.unit" in res["error"]["summary"]

    saved_raw = (raw_dir / "stage.json").read_text(encoding="utf-8")
    if (raw_dir / "request.json").is_file():
        saved_raw += (raw_dir / "request.json").read_text(encoding="utf-8")
    if (raw_dir / "response.json").is_file():
        saved_raw += (raw_dir / "response.json").read_text(encoding="utf-8")
    assert secret not in saved_raw
    assert "data:image/png;base64," not in saved_raw


def test_diagnostics_redact_secrets_in_exception_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    secret = "custom-nonprefix-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    try:
        try:
            raise ValueError(f"key={secret}; Bearer abc.def; data:image/png;base64,YWJjZA==")
        except ValueError as cause:
            raise RuntimeError("failed sk-proj-other-secret") from cause
    except RuntimeError as err:
        details = error_details(err, "тест")
    log_path = tmp_path / "error.log"
    log_error(details, log_path)
    content = json.dumps(details, ensure_ascii=False) + log_path.read_text(encoding="utf-8")
    for forbidden in (secret, "sk-proj-other-secret", "abc.def", "YWJjZA=="):
        assert forbidden not in content
    assert "[СКРЫТО]" in content

    # Проверяем, что отказ записи диагностики не скрывает исходный поток.
    orig_fh = core_logging.file_handler

    def failing_fh(path: Path, **kw: object) -> object:
        if kw.get("diagnostic"):
            raise PermissionError("denied")
        return orig_fh(path, **kw)

    monkeypatch.setattr(core_logging, "file_handler", failing_fh)
    log_error(error_details(ValueError("Primary"), "этап"), tmp_path / "error2.log")
    out = capsys.readouterr().out
    assert "Primary" in out and "Не удалось записать диагностику" in out
