"""Один ответ модели и одно построение; журнал для человека, подробности — в artifacts."""

import asyncio
import base64
import hashlib
import importlib.metadata
import os
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings, RunConfig, RunHooks, Runner
from agents.exceptions import ModelBehaviorError, ModelRefusalError
from agents.models.openai_responses import OpenAIResponsesModel
from agents.usage import Usage
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai.types.shared import Reasoning
from PIL import Image
from pydantic import TypeAdapter, ValidationError

from src.agent.schema import Recipe
from src.config import ROOT, AgentSettings, Config
from src.utils import Run, atomic, error_details, log_error, redact, sha256, write_json

STAGES = {
    "llm": "получение ответа LLM",
    "response_save": "сохранение ответа LLM",
    "recipe": "проверка рецепта",
    "recipe_save": "сохранение рецепта",
    "cad": "построение CAD и экспорт STL",
    "journal": "сохранение журнала",
}


def image_url(source: Path) -> str:
    payload = source.read_bytes()
    with Image.open(BytesIO(payload)) as image:
        if image.format != "PNG":
            raise ValueError("Ожидается PNG")
        image.verify()
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def make_agent(settings: AgentSettings, model: Any = None) -> Agent:
    return Agent(
        name="CAD baseline",
        instructions=settings.prompt,
        model=model or settings.model,
        tools=[],
        output_type=Recipe,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=settings.reasoning_effort),
            max_tokens=settings.max_output_tokens,
            store=False,
        ),
    )


class Journal(RunHooks):
    def __init__(self, run: Run, record: dict) -> None:
        self.run = run
        self.record = record

    async def on_http_response(self, response: Any) -> None:
        """Сохраняем ответ до проверки SDK, в том числе incomplete и ошибки API."""
        await response.aread()
        try:
            payload = response.json()
        except ValueError:
            payload = {"body": response.text}
        self.record["stage"] = "response_save"
        write_json(self.run.artifacts / "response.json", redact(payload))
        info = {"status_code": response.status_code, "request_id": response.headers.get("x-request-id")}
        if isinstance(payload, dict):
            info.update(
                {
                    key: payload[key]
                    for key in ("id", "status", "error", "incomplete_details")
                    if key in payload
                }
            )
            if payload.get("usage"):
                self.record["usage"] = payload["usage"]
        self.record["api_response"] = redact(info)
        write_json(self.run.artifacts / "run.json", self.record)
        self.record["stage"] = "llm"

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self.record["stage"] = "response_save"
        self.record["responses"].append(
            {
                "response_id": response.response_id,
                "request_id": response.request_id,
                "output": redact([item.model_dump(mode="json") for item in response.output]),
            }
        )
        self.record["usage"] = TypeAdapter(Usage).dump_python(context.usage, mode="json")
        write_json(self.run.artifacts / "run.json", self.record)
        self.record["stage"] = "recipe"
        if len(self.record["responses"]) != 1:
            raise RuntimeError("Разрешён ровно один ответ LLM")
        usage = self.record["usage"]
        self.run.log(
            "LLM", f"Ответ получен. Токены: вход {usage['input_tokens']}, выход {usage['output_tokens']}."
        )


def begin(run: Run, kind: str, config: Config) -> dict:
    run.create()
    record = {
        "kind": kind,
        "pair_id": run.pair_id,
        "run_name": run.run_name,
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "started",
        "responses": [],
        "cad_builds": 0,
        "config": config.model_dump(),
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "src").rglob("*.py"))
            if not path.name.startswith("test_")
        },
        "environment_sha256": {name: sha256(ROOT / name) for name in ("pyproject.toml", "uv.lock")},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("openai-agents", "openai", "cadquery", "cadquery-ocp", "trimesh")
        },
    }
    write_json(run.artifacts / "run.json", record)
    run.log("ПРОГОН", run.name)
    return record


def build_once(recipe: Recipe, run: Run, record: dict, config: Config) -> None:
    record["stage"] = "recipe_save"
    write_json(run.directory / "recipe.json", recipe.model_dump())
    record["recipe_sha256"] = sha256(run.directory / "recipe.json")
    record.update(cad_builds=1, stage="cad")
    write_json(run.artifacts / "run.json", record)
    run.log("CAD", f"Строю модель. Операций: {len(recipe.steps)}.")
    from src.agent.cad import build

    record["cad"] = build(recipe, run.directory, config.cad)
    record["status"] = "success"


def record_failure(record: dict, error: BaseException, run: Run) -> None:
    stage = record.get("stage", "подготовка запуска")
    details = error_details(error, STAGES.get(stage, stage))
    api_response = record.get("api_response", {})
    for field in ("status_code", "request_id"):
        if api_response.get(field) is not None:
            details.setdefault(field, api_response[field])
    if api_response.get("error") and isinstance(api_response["error"], dict):
        details.setdefault("code", api_response["error"].get("code"))
    # SDK иногда скрывает причину проверки JSON. Разбираем уже сохранённый текст;
    # запросов и исправлений рецепта здесь нет.
    if isinstance(error, ModelBehaviorError) and stage == "recipe" and record["responses"]:
        output = record["responses"][-1]["output"]
        text = "".join(
            content["text"]
            for item in output
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )
        try:
            Recipe.model_validate_json(text)
        except ValidationError as validation:
            cause = error_details(validation, STAGES["recipe"])
            details["causes"].append({key: cause[key] for key in ("type", "message", "validation")})
            details["traceback"] += "\nПроверка сохранённого ответа:\n" + cause["traceback"]
    if isinstance(error, TimeoutError) and not str(error):
        details["message"] = "Истёк срок ожидания ответа модели"
    interrupted = isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
    if interrupted and not str(error):
        details["message"] = "Запуск прерван"
    record.update(
        status="interrupted" if interrupted else "failed",
        error_type=type(error).__name__,
        error=details,
        failure_kind=(
            "generation"
            if stage in {"llm", "recipe", "cad"}
            and isinstance(error, (ValueError, ModelBehaviorError, ModelRefusalError, TimeoutError))
            and api_response.get("status") != "failed"
            else "infrastructure"
        ),
    )
    log_error(details, run.artifacts / "error.log", run.logfile)


def finish(run: Run, record: dict, started: float) -> None:
    record["seconds"] = time.monotonic() - started
    if record["status"] == "started":
        record.update(status="interrupted", failure_kind="infrastructure")
        run.log("ОШИБКА", "Запуск прерван; полученные файлы сохранены.")
    try:
        write_json(run.artifacts / "run.json", record)
    except OSError as error:
        if record["status"] == "success":
            record["stage"] = "journal"
            record_failure(record, error, run)
        else:
            log_error(
                error_details(error, STAGES["journal"]), run.artifacts / "journal-error.log", run.logfile
            )
    if record["status"] == "success":
        run.log("ГОТОВО", f"STL сохранён. Одно тело, время {record['seconds']:.1f} с.")
    run.log("ФАЙЛЫ", str(run.directory.relative_to(run.root)))


async def generate(source: Path, run: Run, config: Config) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Задайте OPENAI_API_KEY в .env или окружении")
    image = image_url(source)
    input_hash = sha256(source)
    record = begin(run, "single_shot_generation", config)
    record.update(input={"name": source.name, "sha256": input_hash}, stage="llm")
    write_json(run.artifacts / "run.json", record)
    settings = config.agent
    run.log("LLM", f"{settings.model}, рассуждения {settings.reasoning_effort}. Отправляю один запрос.")
    started = time.monotonic()
    journal = Journal(run, record)
    try:
        async with AsyncOpenAI(
            max_retries=0,
            timeout=settings.timeout_seconds,
            base_url="https://api.openai.com/v1",
            http_client=DefaultAsyncHttpxClient(event_hooks={"response": [journal.on_http_response]}),
        ) as client:
            async with asyncio.timeout(settings.timeout_seconds):
                result = await Runner.run(
                    make_agent(settings, OpenAIResponsesModel(model=settings.model, openai_client=client)),
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": image,
                                    "detail": settings.image_detail,
                                }
                            ],
                        }
                    ],
                    max_turns=1,
                    hooks=journal,
                    run_config=RunConfig(tracing_disabled=True),
                )
        if len(record["responses"]) != 1 or not isinstance(result.final_output, Recipe):
            raise ValueError("Единственный ответ не содержит корректного рецепта")
        build_once(result.final_output, run, record, config)
    except (asyncio.CancelledError, KeyboardInterrupt) as error:
        record_failure(record, error, run)
        raise
    except Exception as error:  # noqa: BLE001 — сохраняем исход без повторной генерации.
        record_failure(record, error, run)
    finally:
        finish(run, record, started)
    return record


def replay(source: Path, run: Run, config: Config) -> dict:
    payload = source.read_bytes()  # Рецепт может лежать в перезаписываемой папке.
    record = begin(run, "recipe_replay", config)
    run.log("РЕЦЕПТ", "Читаю сохранённый рецепт. Запросов к LLM нет.")
    started = time.monotonic()
    try:
        record["stage"] = "recipe_save"
        record["source_recipe_sha256"] = hashlib.sha256(payload).hexdigest()
        atomic(run.artifacts / "source_recipe.json", payload)
        record["stage"] = "recipe"
        build_once(Recipe.model_validate_json(payload), run, record, config)
    except KeyboardInterrupt as error:
        record_failure(record, error, run)
        raise
    except Exception as error:  # noqa: BLE001 — сохраняем неудачное построение.
        record_failure(record, error, run)
    finally:
        finish(run, record, started)
    return record
