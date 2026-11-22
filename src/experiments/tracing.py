"""Готовая трассировка Agents SDK в локальный MLflow, связанная с записью попытки."""

import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from src.experiments.logging import log, log_error
from src.experiments.tracking import ensure_run, local_tracking, prepare_tracking, uri
from src.utils import error_details, redact


def sanitize_span(span: Any) -> None:
    """Фильтр MLflow перед экспортом: содержимое PNG и секреты не сохраняются в трассах."""
    from mlflow.entities import SpanStatus

    span.set_inputs(redact(span.inputs))
    span.set_outputs(redact(span.outputs))
    for key, value in span.attributes.items():
        span.set_attribute(key, redact(value))
    for event in span.events:
        for key, value in event.attributes.items():
            event.attributes[key] = redact(value)
    if span.status.description:
        span.set_status(SpanStatus(span.status.status_code, redact(span.status.description)))


@contextmanager
def traced_requests(root: Path, directory: Path, requester: Callable) -> Iterator[Callable]:
    """Инструментирование не меняет сообщения, бюджеты и число обращений к модели."""
    from mlflow.entities import SpanStatus, SpanStatusCode
    from mlflow.tracking import MlflowClient

    context = prepare_tracking(root, directory)
    with local_tracking(root) as mlflow, mlflow.tracing.configure(span_processors=[sanitize_span]):
        client = MlflowClient(tracking_uri=uri(root))
        # Стандартная интеграция убирает экспортёр OpenAI: трассы остаются в локальном MLflow.
        mlflow.openai.autolog(log_traces=True, disable_openai_agent_tracer=True)

        async def request(**kwargs: Any) -> dict:
            raw = kwargs["raw_directory"]
            debug = raw if raw.name == "debug" else raw.parent
            if debug.name != "debug":
                raise ValueError("Диагностика этапа должна лежать в debug попытки")
            destination = debug.parent
            stack = ExitStack()
            try:
                run_id = ensure_run(
                    client, root, directory, destination, context["study"], context["experiment_id"]
                )
                stage = "reading" if destination.parent.name == "readings" else "recipe"
                stack.enter_context(mlflow.start_run(run_id=run_id))
                stack.enter_context(mlflow.set_active_model(model_id=context["model_id"]))
                span = stack.enter_context(mlflow.start_span(name=stage, span_type="CHAIN", run_id=run_id))
                role = (
                    "builder"
                    if stage == "recipe"
                    else ("notes" if "-notes-" in destination.name else "passport")
                )
                snapshot = context["study"]["config"].get("prompt_snapshot", {}).get(role)
                if snapshot:
                    # Штатный API связывает промпт с активным run и ещё не завершённой трассой.
                    mlflow.genai.load_prompt(snapshot["uri"], link_to_model=False, cache_ttl_seconds=0)
                span.set_inputs(
                    {
                        "attempt": destination.relative_to(directory).as_posix(),
                        "stage": stage,
                        "max_output_tokens": kwargs["max_tokens"],
                        "has_png": kwargs["image"] is not None,
                    }
                )
            except Exception as error:  # noqa: BLE001 — сбой истории не блокирует модель.
                close(stack)
                diagnose(error)
                return await requester(**kwargs)
            try:
                result = await requester(**kwargs, tracing=True)
                try:
                    span.set_outputs({key: result.get(key) for key in ("status", "output", "error", "usage")})
                    if result["status"] != "success":
                        span.set_status(
                            SpanStatus(
                                SpanStatusCode.ERROR,
                                (result.get("error") or {}).get("summary", "Этап завершился ошибкой"),
                            )
                        )
                    mlflow.update_current_trace(
                        tags={
                            "study": context["study"]["name"],
                            "stage": stage,
                            "attempt": destination.relative_to(directory).as_posix(),
                            "capture": "live_sdk",
                        }
                    )
                    result["trace_id"] = span.trace_id
                except Exception as error:  # noqa: BLE001 — ответ уже получен, не повторяем запрос.
                    diagnose(error)
                return result
            finally:
                close(stack)

        def diagnose(error: Exception) -> None:
            log_error(
                error_details(error, "трассировка MLflow"),
                root / "logs" / directory.name / "tracing-error.log",
            )

        def close(stack: ExitStack) -> None:
            try:
                stack.__exit__(*sys.exc_info())
            except Exception as error:  # noqa: BLE001 — завершение трассы не подменяет исход запроса.
                diagnose(error)

        try:
            log("ТРАССЫ", "Agents SDK → локальный MLflow. Внешний экспорт трасс отключён.")
            yield request
        finally:
            try:
                mlflow.flush_trace_async_logging()
            except Exception as error:  # noqa: BLE001 — переносимые результаты уже записаны.
                diagnose(error)
            finally:
                mlflow.openai.autolog(disable=True)
