"""Асинхронный параллельный исполнитель матрицы экспериментов с дедупликацией и MLflow."""

import asyncio
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from src.cad.worker import build_recipe
from src.core.config import ExperimentConfig, resolve_prompts
from src.core.logging import log, log_error
from src.core.utils import error_details, require_orientation
from src.evaluation.dataset import check_manifest
from src.experiments.report import evaluate_study, observations, report_study
from src.experiments.store import attempts, case_name, freeze, read_record
from src.llm.client import request as default_request
from src.llm.image import prepare_image
from src.pipelines.base import PipelineContext
from src.pipelines.strategies import get_pipeline


def validate_config_pipelines(config: ExperimentConfig) -> None:
    """Проверяет, что каждый вариант в YAML ссылается на известный pipeline и задаёт нужные промпты."""
    for variant_key, variant in config.variants.items():
        pipeline = get_pipeline(variant.pipeline)
        missing = set(pipeline.required_prompts) - set(variant.prompts)
        if missing:
            req = ", ".join(sorted(missing))
            raise ValueError(f"Вариант {variant_key} (pipeline={variant.pipeline}) требует промпты: {req}")


async def run_matrix(
    root: Path,
    config: ExperimentConfig,
    name: str,
    split: str,
    pairs: list[str] | None,
    repeats: int,
    *,
    concurrency: int | None = None,
    retry_infrastructure: bool = False,
    auto_evaluate: bool = True,
    tracking: bool = True,
    requester: Any = None,
    builder: Any = build_recipe,
) -> Path:
    if requester is None and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Задайте OPENAI_API_KEY в .env или переменных окружения")

    validate_config_pipelines(config)
    config = resolve_prompts(root, config)

    log("ПРОВЕРКА", "Проверяю манифест датасета, SHA256 PNG/STL и разметку ориентации.")
    manifest = check_manifest(
        root,
        config.manifest,
        validation_size=config.execution.validation_size,
        require_val=(split == "val"),
    )
    selected = [
        item
        for item in manifest["items"]
        if item["split"] == split and (not pairs or item["pair_id"] in pairs)
    ]
    if not selected:
        raise ValueError(f"В выборке {split} не найдено ни одной подходящей детали")

    images = {
        item["pair_id"]: prepare_image(
            (root / config.manifest).parent / item["input_png"]["path"],
            config.agent.image_max_side,
        )
        for item in selected
    }
    orientations = {item["pair_id"]: require_orientation(item) for item in selected}

    directory, study = freeze(root, name, config, manifest, split, pairs, repeats)
    effective_concurrency = concurrency or config.execution.concurrency

    log(
        "СЕРИЯ",
        f"{name} · {'отладка' if split == 'dev' else 'валидация'} · "
        f"деталей {len(selected)}, вариантов {len(config.variants)}, повторов {repeats}, "
        f"параллельность {effective_concurrency}.",
    )
    for pair_id, (_, info) in images.items():
        log(
            "PNG",
            f"{pair_id}: {info['original_size']} → {info['sent_size']}, {info['sent_bytes'] / 1024:.0f} КиБ.",
        )

    reading_locks: dict[tuple[str, int, str], asyncio.Lock] = {}
    reading_cache: dict[tuple[str, int, str], tuple[Path, dict]] = {}
    semaphore = asyncio.Semaphore(effective_concurrency)

    with ExitStack() as stack:
        tracker = None
        active_config = config
        stage_requester = requester or default_request
        if tracking:
            try:
                from src.tracking.mlflow import track_study_run

                tracker, active_config = stack.enter_context(track_study_run(root, directory, study))
                stage_requester = tracker.wrap_requester(stage_requester)
            except Exception as error:  # noqa: BLE001 — локальные файлы серии остаются автономными.
                log_error(
                    error_details(error, "инициализация MLflow"),
                    root / "logs" / name / "tracking-error.log",
                )

        total = len(study["schedule"])
        completed_counter = 0

        async def run_cell(index: int, job: dict) -> dict:
            nonlocal completed_counter
            async with semaphore:
                variant_key = job["variant"]
                variant_cfg = active_config.variants[variant_key]
                pipeline = get_pipeline(variant_cfg.pipeline)
                image_b64, image_info = images[job["pair_id"]]
                ctx = PipelineContext(
                    root=root,
                    study_dir=directory,
                    config=active_config,
                    variant_key=variant_key,
                    variant=variant_cfg,
                    job=job,
                    image_b64=image_b64,
                    image_info=image_info,
                    orientation=orientations[job["pair_id"]],
                    retry_infrastructure=retry_infrastructure,
                    requester=stage_requester,
                    builder=builder,
                    reading_locks=reading_locks,
                    reading_cache=reading_cache,
                    tracker=tracker,
                )
                result = await pipeline.execute(ctx)
                completed_counter += 1
                log(
                    "ПРОГРЕСС",
                    f"{completed_counter}/{total} · {variant_key}/{job['pair_id']}-r{job['repeat']:02d} "
                    f"→ {result['status']}",
                )
                return result

        tasks = [asyncio.create_task(run_cell(i, job)) for i, job in enumerate(study["schedule"], 1)]
        await asyncio.gather(*tasks)

    records = [
        read_record(attempts(directory / job["variant"], case_name(job["pair_id"], job["repeat"]))[-1])
        for job in study["schedule"]
    ]
    success = sum(r["status"] == "success" for r in records)
    generation = sum(r.get("failure_kind") == "generation" for r in records)
    infrastructure = sum(r.get("failure_kind") == "infrastructure" for r in records)
    log(
        "ИТОГ СЕРИИ",
        f"Построено {success}/{len(records)}; ошибок генерации {generation}, "
        f"инфраструктуры {infrastructure}.",
    )

    if auto_evaluate:
        evaluate_study(root, name)
        summary = report_study(root, name)
        if tracking:
            try:
                from src.tracking.mlflow import sync_study_results

                rows = observations(root, directory, study)
                sync_study_results(root, directory, study, rows, summary, root / "results" / name)
            except Exception as error:  # noqa: BLE001 — оценка и отчёт уже сохранены в results/.
                log_error(
                    error_details(error, "синхронизация оценок в MLflow"),
                    root / "logs" / name / "tracking-error.log",
                )
    else:
        log("ОЦЕНКА", f"Следующая команда: uv run poe exp evaluate --study {name}")
    return directory
