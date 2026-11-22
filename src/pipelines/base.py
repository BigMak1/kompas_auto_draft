"""Базовый протокол пайплайна и общий исполнитель маршрутов с дедупликацией чтения."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from src.core.config import ExperimentConfig, VariantConfig
from src.core.logging import log, log_error, log_response, scope, waiting
from src.core.utils import atomic, error_details, sha256, write_json
from src.experiments.store import begin_attempt, case_name, save_record
from src.schemas.recipe import Recipe


@dataclass
class PipelineContext:
    root: Path
    study_dir: Path
    config: ExperimentConfig
    variant_key: str
    variant: VariantConfig
    job: dict[str, Any]
    image_b64: str
    image_info: dict[str, Any]
    orientation: str
    retry_infrastructure: bool
    requester: Callable[..., Awaitable[dict]]
    builder: Callable[..., Awaitable[dict]]
    reading_locks: dict[tuple[str, int, str], asyncio.Lock] = field(default_factory=dict)
    reading_cache: dict[tuple[str, int, str], tuple[Path, dict]] = field(default_factory=dict)
    tracker: Any = None

    def prompt_text(self, role: str) -> str:
        rel_path = self.variant.prompts[role]
        return self.config.prompt_snapshot[rel_path]["text"]

    def prompt_uri(self, role: str) -> str | None:
        rel_path = self.variant.prompts.get(role)
        if not rel_path:
            return None
        return self.config.prompt_snapshot[rel_path].get("mlflow_uri")


class Pipeline(Protocol):
    name: str
    required_prompts: tuple[str, ...]
    reading_stage: str | None

    async def execute(self, ctx: PipelineContext) -> dict: ...


def sum_usage(*records: dict | None) -> dict | None:
    if any(
        record is None
        or any(
            type(record.get(key)) is not int or record[key] < 0 for key in ("input_tokens", "output_tokens")
        )
        for record in records
    ):
        return None
    return {
        key: sum(record.get(key, 0) for record in records if record is not None)
        for key in ("input_tokens", "output_tokens")
    }


def make_failure(error: BaseException, stage: str, generation: bool) -> dict:
    return {
        "status": "failed",
        "failure_kind": "generation" if generation else "infrastructure",
        "error": error_details(error, stage),
    }


async def run_shared_reading(
    ctx: PipelineContext,
    *,
    stage: str,
    prompt_role: str,
    output_type: type[BaseModel] | None,
    output_filename: str,
) -> tuple[Path, dict]:
    """Гарантирует ровно 1 LLM-вызов на (pair_id, repeat, stage) даже при параллельном старте."""
    key = (ctx.job["pair_id"], ctx.job["repeat"], stage)
    lock = ctx.reading_locks.setdefault(key, asyncio.Lock())
    async with lock:
        if key in ctx.reading_cache:
            return ctx.reading_cache[key]

        case = f"{case_name(ctx.job['pair_id'], ctx.job['repeat'])}-{stage}"
        destination, existing = begin_attempt(ctx.study_dir / "readings", case, ctx.retry_infrastructure)
        if existing is not None:
            outcome = "успех" if existing["status"] == "success" else "ошибка"
            log(
                "ЧТЕНИЕ",
                f"Сохранённый исход ({stage}): {outcome}, {destination.name}. Запрос не повторяется.",
            )
            ctx.reading_cache[key] = (destination, existing)
            return destination, existing

        started = time.monotonic()
        record: dict[str, Any] = {
            "kind": "reading",
            "stage": stage,
            "pair_id": ctx.job["pair_id"],
            "repeat": ctx.job["repeat"],
            "status": "started",
            "input": ctx.image_info,
            "orientation": ctx.orientation,
        }
        save_record(destination, record)
        technical = destination / "debug"
        logfile = destination / "run.log"
        label = "свободные заметки" if stage == "notes" else "структурированный паспорт"
        instructions = ctx.prompt_text(prompt_role)

        with scope(f"{ctx.job['pair_id']} · чтение ({stage}) · r{ctx.job['repeat']}", logfile=logfile):
            log("ВХОД", f"Читатель получает PNG и подсказку ориентации; результат — {label}.", logfile)
            log(
                "ЧТЕНИЕ",
                f"Отправляю один запрос; лимит {ctx.config.reading.max_output_tokens} выходных токенов.",
                logfile,
            )
            try:
                response = await waiting(
                    ctx.requester(
                        settings=ctx.config.agent,
                        instructions=instructions,
                        image=ctx.image_b64,
                        text=ctx.orientation,
                        output_type=output_type,
                        max_tokens=ctx.config.reading.max_output_tokens,
                        timeout=ctx.config.agent.timeout_seconds,
                        raw_directory=technical,
                        trace_meta={
                            "kind": "reading",
                            "stage": stage,
                            "pair_id": ctx.job["pair_id"],
                            "repeat": ctx.job["repeat"],
                            "attempt": destination.relative_to(ctx.study_dir).as_posix(),
                            "prompt_uri": ctx.prompt_uri(prompt_role),
                        },
                    ),
                    f"ответ LLM — {label}",
                    ctx.config.agent.timeout_seconds,
                    ctx.config.execution.progress_interval_seconds,
                    logfile,
                )
                log_response(response, ctx.config.reading.max_output_tokens, logfile)
                output = response.pop("output", None)
                record.update(response)
                if record["status"] == "success":
                    if record.get("usage") is None:
                        raise RuntimeError("Неизвестен расход токенов на этапе чтения")
                    if record["usage"]["output_tokens"] > ctx.config.reading.max_output_tokens:
                        raise RuntimeError("Этап чтения превысил лимит токенов")
                    if output_type is not None:
                        validated = output_type.model_validate(output)
                        write_json(destination / output_filename, validated.model_dump())
                        if hasattr(validated, "views"):
                            log(
                                "ПАСПОРТ",
                                f"Схема проверена: видов {len(validated.views)}, "
                                f"элементов {len(validated.features)}, размеров {len(validated.dimensions)}.",
                                logfile,
                            )
                    else:
                        if not isinstance(output, str) or not output.strip():
                            raise ValueError("Пустые заметки")
                        atomic(destination / output_filename, output.encode("utf-8"))
                        log("ЗАМЕТКИ", f"Сохранены: {len(output)} символов.", logfile)
            except (KeyboardInterrupt, asyncio.CancelledError) as error:
                record.update(make_failure(error, "чтение чертежа", False))
                raise
            except Exception as error:  # noqa: BLE001 — фиксируем сбой этапа чтения.
                record.update(
                    make_failure(error, "чтение чертежа", isinstance(error, (ValueError, TimeoutError)))
                )
            finally:
                record["seconds"] = max(record.get("seconds", 0.0), time.monotonic() - started)
                if record.get("error"):
                    log_error(record["error"], technical / "error.log", logfile)
                save_record(destination, record, final=True)
        ctx.reading_cache[key] = (destination, record)
        return destination, record


async def execute_two_or_one_stage(
    ctx: PipelineContext,
    *,
    reading_stage: str | None = None,
    reading_prompt_role: str | None = None,
    reading_output_type: type[BaseModel] | None = None,
    reading_filename: str | None = None,
    builder_sees_image: bool = True,
) -> dict:
    """Общий поток выполнения маршрута (1 или 2 этапа + CadQuery) с учётом бюджетов."""
    case = case_name(ctx.job["pair_id"], ctx.job["repeat"])
    destination, existing = begin_attempt(ctx.study_dir / ctx.variant_key, case, ctx.retry_infrastructure)
    if existing is not None:
        outcome = "успех" if existing["status"] == "success" else "ошибка"
        log("ПРОПУСК", f"{ctx.variant.label} / {case}: сохранённый исход — {outcome}.")
        return existing

    technical = destination / "debug"
    logfile = destination / "run.log"
    record: dict[str, Any] = {
        **ctx.job,
        "kind": "route",
        "status": "started",
        "stage": "preparation",
        "input": ctx.image_info,
        "orientation": ctx.orientation,
        "seconds": 0.0,
        "cad_builds": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    save_record(destination, record)

    started: float | None = time.monotonic()
    reading_seconds = 0.0
    reading_usage: dict | None = {"input_tokens": 0, "output_tokens": 0}
    own_usage: dict | None = {"input_tokens": 0, "output_tokens": 0}
    scope_label = f"{ctx.job['pair_id']} · {ctx.variant_key} · r{ctx.job['repeat']}"

    with scope(scope_label, logfile=logfile):
        log("МАРШРУТ", f"{ctx.variant.label} · начинаю.", logfile)
        trace_span = None
        trace_ctx = None
        if ctx.tracker is not None:
            trace_ctx = ctx.tracker.start_trial_trace(
                variant_key=ctx.variant_key,
                pair_id=ctx.job["pair_id"],
                repeat=ctx.job["repeat"],
                attempt=destination.relative_to(ctx.study_dir).as_posix(),
                has_png=builder_sees_image,
            )
            trace_span = trace_ctx.__enter__()
            if trace_span is not None:
                record["trace_id"] = trace_span.trace_id
        try:
            text = ctx.orientation
            if reading_stage is not None:
                assert reading_prompt_role and reading_filename
                record["stage"] = "reading"
                source, extracted = await run_shared_reading(
                    ctx,
                    stage=reading_stage,
                    prompt_role=reading_prompt_role,
                    output_type=reading_output_type,
                    output_filename=reading_filename,
                )
                # После ожидания/выполнения чтения считаем бюджет по сохранённому времени чтения.
                started = time.monotonic()
                reading_seconds = extracted.get("seconds", 0.0)
                reading_usage = extracted.get("usage")
                record["reading_path"] = source.relative_to(ctx.study_dir).as_posix()
                record["reading_sha256"] = sha256(source / "record.json")
                if extracted.get("trace_id"):
                    record["reading_trace_id"] = extracted["trace_id"]
                if extracted["status"] != "success":
                    record.update(
                        status="failed",
                        failure_kind=extracted.get("failure_kind", "infrastructure"),
                        error=extracted.get(
                            "error", error_details(RuntimeError("Чтение было прервано"), "чтение")
                        ),
                        error_source=record["reading_path"],
                        failure_reason=extracted.get("failure_reason"),
                    )
                    return record
                description = (source / reading_filename).read_text(encoding="utf-8")
                atomic(destination / reading_filename, description.encode("utf-8"))
                text += "\n\n" + description

            if reading_usage is None:
                raise RuntimeError("Бюджет чтения неизвестен")
            tokens = ctx.config.agent.max_output_tokens - reading_usage["output_tokens"]
            remaining = ctx.config.agent.timeout_seconds - reading_seconds
            if tokens <= 0 or remaining <= 0:
                raise TimeoutError("Бюджет маршрута исчерпан до этапа построения рецепта")

            record["stage"] = "recipe"
            record["llm_started"] = True
            own_usage = None
            save_record(destination, record)

            input_desc = (
                "PNG + подсказка ориентации"
                if reading_stage is None
                else (
                    f"{reading_stage} + PNG + подсказка ориентации"
                    if builder_sees_image
                    else f"{reading_stage} + подсказка ориентации (PNG скрыт)"
                )
            )
            log("ВХОД", f"Построитель: {input_desc}.", logfile)
            log(
                "РЕЦЕПТ",
                f"Отправляю один запрос. Осталось {tokens} выходных токенов и {remaining:.1f} с.",
                logfile,
            )
            response = await waiting(
                ctx.requester(
                    settings=ctx.config.agent,
                    instructions=ctx.prompt_text("builder"),
                    image=ctx.image_b64 if builder_sees_image else None,
                    text=text,
                    output_type=Recipe,
                    max_tokens=tokens,
                    timeout=remaining,
                    raw_directory=technical / "llm",
                    trace_meta={
                        "kind": "recipe",
                        "variant": ctx.variant_key,
                        "pair_id": ctx.job["pair_id"],
                        "repeat": ctx.job["repeat"],
                        "attempt": destination.relative_to(ctx.study_dir).as_posix(),
                        "prompt_uri": ctx.prompt_uri("builder"),
                        "reading_trace_id": record.get("reading_trace_id"),
                    },
                ),
                "ответ LLM — рецепт CAD",
                remaining,
                ctx.config.execution.progress_interval_seconds,
                logfile,
            )
            log_response(response, tokens, logfile)
            output = response.pop("output", None)
            record["llm"] = response
            own_usage = response.get("usage")
            if response["status"] != "success":
                record.update({key: response[key] for key in ("status", "failure_kind", "error")})
                record["failure_reason"] = response.get("failure_reason")
                return record
            if own_usage is None or not 0 <= own_usage["output_tokens"] <= tokens:
                raise RuntimeError("Расход токенов построителя отсутствует или превышает лимит")

            recipe = Recipe.model_validate(output)
            write_json(destination / "recipe.json", recipe.model_dump())
            log("РЕЦЕПТ", f"Проверен и сохранён: {len(recipe.steps)} операций.", logfile)

            record["stage"] = "cad"
            remaining = ctx.config.agent.timeout_seconds - reading_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Бюджет маршрута исчерпан до запуска CadQuery")
            record["cad_builds"] = 1
            save_record(destination, record)
            log("CAD", f"Запускаю CadQuery и экспорт STL; осталось {remaining:.1f} с.", logfile)

            if ctx.tracker is not None:
                cad_result = await ctx.tracker.trace_cad_step(
                    waiting(
                        ctx.builder(
                            recipe.model_dump(),
                            destination,
                            technical / "cad",
                            ctx.config.cad,
                            remaining,
                        ),
                        "построение CAD и экспорт STL",
                        remaining,
                        ctx.config.execution.progress_interval_seconds,
                        logfile,
                    ),
                    steps=len(recipe.steps),
                )
            else:
                cad_result = await waiting(
                    ctx.builder(
                        recipe.model_dump(),
                        destination,
                        technical / "cad",
                        ctx.config.cad,
                        remaining,
                    ),
                    "построение CAD и экспорт STL",
                    remaining,
                    ctx.config.execution.progress_interval_seconds,
                    logfile,
                )
            record.update(cad_result)
            if reading_seconds + (time.monotonic() - started) > ctx.config.agent.timeout_seconds:
                raise TimeoutError("Общий лимит времени маршрута исчерпан")
            if record["status"] == "success":
                log("ГОТОВО", "STL построен, одно замкнутое тело.", logfile)
                noop_steps = (record.get("cad") or {}).get("noop_steps") or []
                if noop_steps:
                    log("ВНИМАНИЕ", f"Шаги без изменения объёма (перекрытие): {noop_steps}.", logfile)
        except (KeyboardInterrupt, asyncio.CancelledError) as error:
            record.update(make_failure(error, record["stage"], False))
            raise
        except Exception as error:  # noqa: BLE001 — исход ячейки всегда сохраняется на диск.
            record.update(make_failure(error, record["stage"], isinstance(error, (ValueError, TimeoutError))))
        finally:
            record["seconds"] = reading_seconds + (time.monotonic() - started if started is not None else 0.0)
            record["usage"] = sum_usage(reading_usage, own_usage)
            if record.get("error"):
                if record.get("error_source"):
                    reason = record["error"].get("summary", record["error"]["message"])
                    log("ОСТАНОВЛЕН", f"Чтение не удалось ({reading_stage}). {reason}", logfile)
                    log("ИСТОЧНИК", f"{record['error_source']}; этап рецепта и CAD не запускались.", logfile)
                else:
                    log_error(record["error"], technical / "error.log", logfile)
            log(
                "ИТОГ",
                f"{'Успех' if record['status'] == 'success' else 'Неудачный исход'} · "
                f"{record['seconds']:.1f} с · построений {record['cad_builds']}.",
                logfile,
            )
            if trace_ctx is not None:
                ctx.tracker.finish_trial_trace(trace_ctx, trace_span, record)
            save_record(destination, record, final=True)
    return record
