"""Сравнение маршрутов: фиксированная матрица, отдельные контексты и общий бюджет."""

import asyncio
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from src.agent.schemas.passport import Passport
from src.agent.schemas.recipe import Recipe
from src.config import ROOT
from src.experiments.config import LABELS, ExperimentConfig, check_manifest
from src.experiments.logging import log, log_error, log_response, scope, waiting
from src.experiments.store import (
    attempts,
    begin_attempt,
    case_name,
    freeze,
    read_record,
    save_record,
)
from src.utils import (
    atomic,
    error_details,
    read_json,
    redact,
    require_orientation,
    sha256,
    write_json,
)


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


def failure(error: BaseException, stage: str, generation: bool) -> dict:
    return {
        "status": "failed",
        "failure_kind": "generation" if generation else "infrastructure",
        "error": error_details(error, stage),
    }


async def build_recipe(
    recipe: dict, directory: Path, technical: Path, config: ExperimentConfig, timeout: float
) -> dict:
    """Завершаем только собственный процесс CadQuery; внешние приложения не участвуют."""
    write_json(technical / "build-input.json", {"recipe": recipe, "settings": config.cad.model_dump()})
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "src.agent.build_worker",
        str(technical.resolve()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
        cwd=ROOT,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    atomic(technical / "cad.log", redact((stdout + stderr).decode(errors="replace")).encode())
    result_path = technical / "build-result.json"
    if not result_path.is_file():
        raise RuntimeError(f"Процесс CAD завершился без результата, код {process.returncode}")
    result = read_json(result_path)
    if result["status"] == "success":
        atomic(directory / "model.stl", (technical / "model.stl").read_bytes())
        for name in ("model.stl", "build-input.json", "build-result.json"):
            (technical / name).unlink(missing_ok=True)
    return result


async def reading(
    root: Path,
    directory: Path,
    config: ExperimentConfig,
    job: dict,
    image: str,
    image_info: dict,
    orientation: str,
    retry: bool,
    requester: Any,
) -> tuple[Path, dict]:
    stage = "notes" if job["variant"] == "notes_png" else "passport"
    case = case_name(job["pair_id"], job["repeat"]) + "-" + stage
    destination, existing = begin_attempt(directory / "readings", case, retry)
    if existing is not None:
        outcome = "успех" if existing["status"] == "success" else "ошибка"
        label = "заметки" if stage == "notes" else "общий паспорт"
        log(
            "ЧТЕНИЕ",
            f"Сохранённый исход ({label}): {outcome}, {destination.name}. Новый запрос не отправляется.",
        )
        return destination, existing
    started = time.monotonic()
    record = {
        "kind": "reading",
        "stage": stage,
        "pair_id": job["pair_id"],
        "repeat": job["repeat"],
        "status": "started",
        "input": image_info,
        "orientation": orientation,
    }
    save_record(destination, record)
    technical = destination / "debug"
    instructions = (
        (config.reading.prompt + "\n" if config.reading.prompt else "")
        + ""
        + (config.reading.notes_format if stage == "notes" else config.reading.passport_format)
    )
    label = "свободные заметки" if stage == "notes" else "структурированный паспорт"
    logfile = destination / "run.log"
    log("ВХОД", f"Читатель получает PNG и подсказку ориентации; результат — {label}.", logfile)
    log(
        "ЧТЕНИЕ",
        f"Отправляю один запрос; лимит {config.reading.max_output_tokens} выходных токенов. Жду ответ LLM.",
        logfile,
    )
    try:
        response = await waiting(
            requester(
                settings=config.agent,
                instructions=instructions,
                image=image,
                text=orientation,
                output_type=None if stage == "notes" else Passport,
                max_tokens=config.reading.max_output_tokens,
                timeout=config.agent.timeout_seconds,
                raw_directory=technical,
            ),
            f"ответ LLM — {label}",
            config.agent.timeout_seconds,
            config.progress_interval_seconds,
            logfile,
        )
        log_response(response, config.reading.max_output_tokens, logfile)
        output = response.pop("output", None)
        record.update(response)
        if record["status"] == "success":
            if record.get("usage") is None:
                raise RuntimeError("Неизвестен расход первого этапа")
            if record["usage"]["output_tokens"] > config.reading.max_output_tokens:
                raise RuntimeError("Первый этап превысил бюджет")
            if stage == "passport":
                passport = Passport.model_validate(output)
                write_json(destination / "passport.json", passport.model_dump())
                log(
                    "ПАСПОРТ",
                    f"Схема и ссылки проверены: видов {len(passport.views)}, "
                    f"элементов {len(passport.features)}, размеров {len(passport.dimensions)}.",
                    logfile,
                )
            else:
                if not isinstance(output, str) or not output.strip():
                    raise ValueError("Пустые заметки")
                atomic(destination / "notes.md", output.encode())
                log("ЗАМЕТКИ", f"Сохранены: {len(output)} символов.", logfile)
    except (KeyboardInterrupt, asyncio.CancelledError) as error:
        record.update(failure(error, "чтение чертежа", False))
        raise
    except Exception as error:  # noqa: BLE001 — сохраняем в том числе ошибки файловой системы.
        record.update(failure(error, "чтение чертежа", isinstance(error, (ValueError, TimeoutError))))
    finally:
        record["seconds"] = max(record.get("seconds", 0.0), time.monotonic() - started)
        if record.get("error"):
            log_error(record["error"], technical / "error.log", destination / "run.log")
        save_record(destination, record, final=True)
    return destination, record


async def route(
    root: Path,
    directory: Path,
    config: ExperimentConfig,
    job: dict,
    image: str,
    image_info: dict,
    orientation: str,
    retry: bool,
    requester: Any,
    builder: Any,
) -> dict:
    case = case_name(job["pair_id"], job["repeat"])
    destination, existing = begin_attempt(directory / job["variant"], case, retry)
    if existing is not None:
        outcome = "успех" if existing["status"] == "success" else "ошибка"
        log("ПРОПУСК", f"{LABELS[job['variant']]} / {case}: сохранённый исход — {outcome}.")
        return existing
    technical = destination / "debug"
    logfile = destination / "run.log"
    record: dict[str, Any] = {
        **job,
        "kind": "route",
        "status": "started",
        "stage": "preparation",
        "input": image_info,
        "orientation": orientation,
        "seconds": 0.0,
        "cad_builds": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    save_record(destination, record)
    started: float | None = time.monotonic()
    reading_seconds = 0.0
    reading_usage: dict | None = {"input_tokens": 0, "output_tokens": 0}
    own_usage: dict | None = {"input_tokens": 0, "output_tokens": 0}
    log("МАРШРУТ", f"{LABELS[job['variant']]} · начинаю.", logfile)
    try:
        text = orientation
        if job["variant"] != "direct":
            record["stage"] = "reading"
            source, extracted = await reading(
                root, directory, config, job, image, image_info, orientation, retry, requester
            )
            started = time.monotonic()
            reading_seconds = extracted.get("seconds", 0.0)
            reading_usage = extracted.get("usage")
            record.update(
                reading_path=source.relative_to(directory).as_posix(),
                reading_sha256=sha256(source / "record.json"),
            )
            if extracted["status"] != "success":
                record.update(status="failed", failure_kind=extracted.get("failure_kind", "infrastructure"))
                record["error"] = extracted.get(
                    "error", error_details(RuntimeError("Чтение было прервано"), "чтение")
                )
                record["error_source"] = record["reading_path"]
                record["failure_reason"] = extracted.get("failure_reason")
                return record
            filename = "notes.md" if job["variant"] == "notes_png" else "passport.json"
            description = (source / filename).read_text(encoding="utf-8")
            atomic(destination / filename, description.encode())
            # Ориентация задана человеком и передаётся независимо от ответа читателя.
            text += "\n\n" + description
        if reading_usage is None:
            raise RuntimeError("Бюджет чтения неизвестен")
        tokens = config.agent.max_output_tokens - reading_usage["output_tokens"]
        remaining = config.agent.timeout_seconds - reading_seconds
        if tokens <= 0 or remaining <= 0:
            raise TimeoutError("Бюджет исчерпан до построителя рецепта")
        record["stage"] = "recipe"
        record["llm_started"] = True
        own_usage = None
        save_record(destination, record)
        inputs = {
            "direct": "PNG + подсказка ориентации",
            "passport_only": "паспорт + подсказка ориентации; PNG отсутствует",
            "passport_png": "паспорт + PNG + подсказка ориентации",
            "notes_png": "заметки + PNG + подсказка ориентации",
        }
        log("ВХОД", "Построитель: " + inputs[job["variant"]] + ".", logfile)
        log(
            "РЕЦЕПТ",
            f"Отправляю один запрос. Осталось {tokens} выходных токенов и {remaining:.1f} с. Жду ответ LLM.",
            logfile,
        )
        response = await waiting(
            requester(
                settings=config.agent,
                instructions=config.agent.prompt,
                image=None if job["variant"] == "passport_only" else image,
                text=text,
                output_type=Recipe,
                max_tokens=tokens,
                timeout=remaining,
                raw_directory=technical / "llm",
            ),
            "ответ LLM — рецепт CAD",
            remaining,
            config.progress_interval_seconds,
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
            raise RuntimeError("Расход построителя отсутствует или превышает лимит")
        recipe = Recipe.model_validate(output)
        write_json(destination / "recipe.json", recipe.model_dump())
        log("РЕЦЕПТ", f"Проверен и сохранён: {len(recipe.steps)} операций.", logfile)
        record["stage"] = "cad"
        remaining = config.agent.timeout_seconds - reading_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Бюджет исчерпан до CadQuery")
        record["cad_builds"] = 1
        save_record(destination, record)
        log("CAD", f"Начинаю построение CadQuery и экспорт STL; осталось {remaining:.1f} с.", logfile)
        result = await waiting(
            builder(recipe.model_dump(), destination, technical / "cad", config, remaining),
            "построение CAD и экспорт STL",
            remaining,
            config.progress_interval_seconds,
            logfile,
        )
        record.update(result)
        if reading_seconds + time.monotonic() - started > config.agent.timeout_seconds:
            raise TimeoutError("Общий бюджет маршрута исчерпан")
        if record["status"] == "success":
            log("ГОТОВО", "STL построен, одно тело", logfile)
    except (KeyboardInterrupt, asyncio.CancelledError) as error:
        record.update(failure(error, record["stage"], False))
        raise
    except Exception as error:  # noqa: BLE001 — исход каждой ячейки матрицы сохраняется.
        record.update(failure(error, record["stage"], isinstance(error, (ValueError, TimeoutError))))
    finally:
        record["seconds"] = reading_seconds + (time.monotonic() - started if started is not None else 0.0)
        record["usage"] = sum_usage(reading_usage, own_usage)
        if record.get("error"):
            if record.get("error_source"):
                reason = record["error"].get("summary", record["error"]["message"])
                label = "заметки" if job["variant"] == "notes_png" else "общий паспорт"
                log("ОСТАНОВЛЕН", f"Не удалось прочитать чертёж ({label}). {reason}", logfile)
                log(
                    "ИСТОЧНИК",
                    f"{record['error_source']}; построитель рецепта и CAD не запускались.",
                    logfile,
                )
            else:
                log_error(record["error"], technical / "error.log", logfile)
        log(
            "ИТОГ",
            f"{'Успех' if record['status'] == 'success' else 'Неудачный исход'} · "
            f"{record['seconds']:.1f} с · построений {record['cad_builds']}.",
            logfile,
        )
        save_record(destination, record, final=True)
    return record


async def run_matrix(
    root: Path,
    config: ExperimentConfig,
    name: str,
    split: str,
    pairs: list[str] | None,
    repeats: int,
    *,
    retry: bool = False,
    requester: Any = None,
    builder: Any = build_recipe,
    tracking: bool = True,
) -> Path:
    from src.agent.agent import prepare_image
    from src.agent.stages import request

    if requester is None and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Задайте OPENAI_API_KEY в .env или окружении")
    log("ПРОВЕРКА", "Проверяю конфигурацию, хеши PNG/STL и разметку ориентации.")
    manifest = check_manifest(root, config, require_val=split == "val")
    selected = [
        item
        for item in manifest["items"]
        if item["split"] == split and (not pairs or item["pair_id"] in pairs)
    ]
    # Проверяем каждый PNG до первого обращения. Эталоны не входят в подготовленные сообщения.
    images = {
        item["pair_id"]: prepare_image(
            (root / config.manifest).parent / item["input_png"]["path"], config.agent.image_max_side
        )
        for item in selected
    }
    orientations = {item["pair_id"]: require_orientation(item) for item in selected}
    from src.experiments.prompts import resolve_prompts

    config = resolve_prompts(root, config)
    directory, study = freeze(root, name, config, manifest, split, pairs, repeats)
    if tracking:
        from src.experiments.tracking import prepare_tracking

        # Ошибка закрепления версии останавливает генерацию до первого запроса.
        prepare_tracking(root, directory, restore_deleted=True)
    log(
        "СЕРИЯ",
        f"{name} · {'отладка' if split == 'dev' else 'валидация'} · деталей {len(selected)}, "
        f"маршрутов {len(config.variants)}, повторов {repeats}.",
    )
    readings_per_pair = int(any(v in config.variants for v in ("passport_only", "passport_png"))) + int(
        "notes_png" in config.variants
    )
    requests = len(selected) * repeats * (len(config.variants) + readings_per_pair)
    log(
        "ПЛАН",
        f"Результатов {len(study['schedule'])}; до {requests} запросов LLM. "
        "Общий паспорт извлекается один раз на деталь и повтор.",
    )
    for pair, (_, info) in images.items():
        log(
            "PNG",
            f"{pair}: {info['original_size']} → {info['sent_size']}, "
            f"{info['sent_bytes'] / 1024:.0f} КиБ. Подсказка ориентации добавлена.",
        )
    log("СНИМОК", "Конфигурация, входы и код зафиксированы; сохранённые исходы не перезаписываются.")
    with ExitStack() as tracing_context:
        stage_requester = requester or request
        if tracking:
            try:
                from src.experiments.tracing import traced_requests

                stage_requester = tracing_context.enter_context(
                    traced_requests(root, directory, stage_requester)
                )
            except Exception as error:  # noqa: BLE001 — исходы остаются переносимыми при сбое MLflow.
                log_error(
                    error_details(error, "подготовка трассировки MLflow"),
                    root / "logs" / name / "tracking-error.log",
                )
        for index, job in enumerate(study["schedule"], 1):
            log("СЕРИЯ", f"{index}/{len(study['schedule'])} · {name}")
            image, info = images[job["pair_id"]]
            with scope(f"{job['pair_id']} · {LABELS[job['variant']]} · повтор {job['repeat']}/{repeats}"):
                await route(
                    root,
                    directory,
                    config,
                    job,
                    image,
                    info,
                    orientations[job["pair_id"]],
                    retry,
                    stage_requester,
                    builder,
                )
            if tracking:
                try:
                    from src.experiments.tracking import sync_study

                    log("MLFLOW", "Сохраняю исход в локальную историю.")
                    sync_study(directory, root)
                except Exception as error:  # noqa: BLE001 — MLflow можно восстановить из переносимых записей.
                    log_error(error_details(error, "MLflow"), root / "logs" / name / "tracking-error.log")
    records = [
        read_record(attempts(directory / job["variant"], case_name(job["pair_id"], job["repeat"]))[-1])
        for job in study["schedule"]
    ]
    success = sum(record["status"] == "success" for record in records)
    generation = sum(record.get("failure_kind") == "generation" for record in records)
    infrastructure = sum(record.get("failure_kind") == "infrastructure" for record in records)
    log(
        "ИТОГ СЕРИИ",
        f"Построено {success}/{len(records)}; ошибок генерации {generation}, "
        f"инфраструктуры {infrastructure}.",
    )
    log("ОЦЕНКА", f"Следующая команда: uv run poe experiment evaluate --study {name}")
    return directory
