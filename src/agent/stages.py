"""Один изолированный запрос SDK. Ни файловой системы, ни эталонов у модели нет."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings, RunConfig, Runner
from agents.exceptions import ModelBehaviorError, ModelRefusalError
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai.types.shared import Reasoning
from pydantic import BaseModel, ValidationError

from src.config import AgentSettings
from src.utils import error_details, redact, write_json


def messages(image: str | None, text: str | None, detail: str) -> list[dict]:
    """Формируем новый контекст по белому списку; историю предыдущего запроса не переносим."""
    content = []
    if text is not None:
        content.append({"type": "input_text", "text": text})
    if image is not None:
        content.append({"type": "input_image", "image_url": image, "detail": detail})
    if not content:
        raise ValueError("У этапа нет входных данных")
    return [{"role": "user", "content": content}]


def diagnose_output(payload: dict, output_type: type[BaseModel] | None) -> dict | None:
    """SDK скрывает ValidationError. Проверяем сохранённый текст, не исправляя его."""
    if output_type is None:
        return None
    text = "".join(
        content.get("text", "")
        for item in payload.get("output", [])
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    )
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as error:
        return {**error_details(error, "синтаксис JSON"), "code": "invalid_json"}
    try:
        output_type.model_validate(value)
    except ValidationError as error:
        details = error_details(error, "проверка схемы ответа")
        details["code"] = (
            "invalid_references"
            if any(item["type"] == "passport_reference" for item in details["validation"])
            else "schema_validation"
        )
        details["summary"] = "; ".join(
            f"{'.'.join(map(str, item['loc'])) or 'JSON'}: {item['msg']}" for item in details["validation"]
        )
        return details
    return None


async def request(
    *,
    settings: AgentSettings,
    instructions: str,
    image: str | None,
    text: str | None,
    output_type: type[BaseModel] | None,
    max_tokens: int,
    timeout: float,
    raw_directory: Path,
    tracing: bool = False,
) -> dict:
    """Возвращаем и сохраняем расход даже при ошибке проверки структурированного ответа."""
    started = time.monotonic()
    record: dict[str, Any] = {"status": "started", "usage": None, "http_responses": 0}
    captured: dict = {}

    async def capture_request(request: Any) -> None:
        write_json(raw_directory / "request.json", redact(json.loads(request.content)))

    async def capture(response: Any) -> None:
        nonlocal captured
        await response.aread()
        try:
            payload = response.json()
        except ValueError:
            payload = {"body": response.text}
        write_json(raw_directory / "response.json", redact(payload))
        record["http_responses"] += 1
        record["request_id"] = response.headers.get("x-request-id")
        record["http_status"] = response.status_code
        if isinstance(payload, dict):
            captured = payload
            record["response_id"] = payload.get("id")
            record["response_status"] = payload.get("status")
            record["returned_model"] = payload.get("model")
            record["usage"] = payload.get("usage")
            record["api_error"] = redact(payload.get("error"))
            record["incomplete_details"] = payload.get("incomplete_details")

    try:
        async with AsyncOpenAI(
            max_retries=0,
            timeout=timeout,
            base_url="https://api.openai.com/v1",
            http_client=DefaultAsyncHttpxClient(
                event_hooks={"request": [capture_request], "response": [capture]}
            ),
        ) as client:
            agent = Agent(
                name="CAD experiment stage",
                instructions=instructions,
                model=OpenAIResponsesModel(model=settings.model, openai_client=client),
                tools=[],
                output_type=output_type,
                model_settings=ModelSettings(
                    reasoning=Reasoning(effort=settings.reasoning_effort), max_tokens=max_tokens, store=False
                ),
            )
            result = await asyncio.wait_for(
                Runner.run(
                    agent,
                    input=messages(image, text, settings.image_detail),
                    max_turns=1,
                    run_config=RunConfig(tracing_disabled=not tracing),
                ),
                timeout=timeout,
            )
        if record["http_responses"] != 1:
            raise RuntimeError("Этап должен выполнить ровно один запрос")
        usage = record["usage"]
        if not isinstance(usage, dict) or any(
            type(usage.get(key)) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens")
        ):
            raise RuntimeError("API не вернул расход: продолжать маршрут нельзя")
        if not 0 <= usage["output_tokens"] <= max_tokens:
            raise RuntimeError("Ответ превысил установленный бюджет")
        output = result.final_output
        if output_type is not None:
            if not isinstance(output, output_type):
                raise ValueError("Ответ не соответствует модели этапа")
            record["output"] = output.model_dump(mode="json")
        else:
            if not isinstance(output, str) or not output.strip():
                raise ValueError("Пустые заметки")
            record["output"] = output
        record["status"] = "success"
    except Exception as error:  # noqa: BLE001 — ошибка должна сохраниться до возврата раннеру.
        generation = (
            isinstance(
                error, (ValueError, ValidationError, ModelBehaviorError, ModelRefusalError, TimeoutError)
            )
            and record.get("response_status") != "failed"
        )
        record.update(
            status="failed",
            failure_kind="generation" if generation else "infrastructure",
            error=error_details(error, "ответ LLM / проверка схемы"),
        )
        record["error"]["code"] = "api_error" if not generation else "model_output"
        if isinstance(error, ModelRefusalError):
            record["error"]["code"] = "refusal"
        elif (
            isinstance(error, (ModelBehaviorError, ValidationError)) and captured.get("status") == "completed"
        ):
            details = diagnose_output(captured, output_type)
            if details:
                details["sdk_error"] = record["error"]
                record["error"] = details
        for source, target in (("request_id", "request_id"), ("http_status", "status_code")):
            if record.get(source) is not None:
                record["error"][target] = record[source]
        if record.get("incomplete_details"):
            record["error"]["code"] = "incomplete_response"
            record["error"]["notes"] = [f"Незавершённый ответ: {record['incomplete_details']}"]
        if (record.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
            record["failure_reason"] = "max_output_tokens"
            record["error"]["code"] = "max_output_tokens"
            record["error"]["stage"] = "получение полного ответа LLM"
            usage = record.get("usage") or {}
            reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            detail = f", из них {reasoning} на рассуждения" if reasoning is not None else ""
            record["error"]["summary"] = (
                f"Ответ обрезан по лимиту {max_tokens} выходных токенов{detail}. "
                "Полный ответ не получен; повторного запроса нет."
            )
    finally:
        record["seconds"] = time.monotonic() - started
        write_json(raw_directory / "stage.json", redact(record))
    return redact(record)
