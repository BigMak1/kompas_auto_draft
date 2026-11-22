"""Офлайн-тесты параллельного раннера, дедупликации паспорта, бюджетов и resume."""

import asyncio
import json
from pathlib import Path

import pytest

from src.core.config import ExperimentConfig
from src.core.utils import read_json
from src.experiments.report import aggregate, evaluate_study, observations
from src.experiments.runner import run_matrix
from src.experiments.store import read_record, verify_study
from src.schemas.passport import Passport
from src.schemas.recipe import Recipe
from tests.conftest import PASSPORT, FakePipelineModel, fake_cad_builder


def test_parallel_matrix_deduplicates_shared_passport_and_enforces_budgets(
    sandbox: tuple[Path, ExperimentConfig],
) -> None:
    root, cfg = sandbox
    # Задержка заставляет все 4 задачи стартовать одновременно внутри Semaphore(4).
    model = FakePipelineModel(delay=0.05)
    directory = asyncio.run(
        run_matrix(
            root,
            cfg,
            "parallel-unit",
            "dev",
            None,
            repeats=3,
            concurrency=4,
            tracking=False,
            auto_evaluate=False,
            requester=model,
            builder=fake_cad_builder,
        )
    )
    # 3 повтора * (4 построителя + 1 общий паспорт + 1 заметки) = 18 запросов вместо 21!
    assert len(model.calls) == 18
    assert sum(c["output_type"] is Passport for c in model.calls) == 3
    assert sum(c["output_type"] is None for c in model.calls) == 3
    builders = [c for c in model.calls if c["output_type"] is Recipe]
    assert len(builders) == 12
    assert sum(c["image"] is None for c in builders) == 3  # только 03_passport_only не видит PNG
    # Проверяем вычитание бюджета токенов чтения (900) из общего лимита (16384 -> 15484).
    assert sum(c["max_tokens"] == 16384 for c in builders) == 3  # direct
    assert sum(c["max_tokens"] == 15484 for c in builders) == 9  # 2-stage routes

    for repeat in range(1, 4):
        only = read_record(directory / "03_passport_only" / f"DEV1-r{repeat:02d}-a001")
        png = read_record(directory / "02_passport_png" / f"DEV1-r{repeat:02d}-a001")
        assert only["reading_path"] == png["reading_path"]
        assert only["reading_sha256"] == png["reading_sha256"]
        assert only["usage"]["output_tokens"] == 1600
        assert json.loads((directory / only["reading_path"] / "passport.json").read_text()) == PASSPORT

    # Повторный запуск (resume) не делает ни одного нового запроса.
    asyncio.run(
        run_matrix(
            root,
            cfg,
            "parallel-unit",
            "dev",
            None,
            repeats=3,
            concurrency=4,
            tracking=False,
            auto_evaluate=False,
            requester=model,
            builder=fake_cad_builder,
        )
    )
    assert len(model.calls) == 18


def test_expired_reading_budget_stops_builder_and_cad(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, cfg = sandbox
    model = FakePipelineModel(reading_seconds=905.0)
    directory = asyncio.run(
        run_matrix(
            root,
            cfg,
            "expired-budget",
            "dev",
            None,
            repeats=1,
            concurrency=4,
            tracking=False,
            auto_evaluate=False,
            requester=model,
            builder=fake_cad_builder,
        )
    )
    assert len(model.calls) == 3  # direct + 1 passport + 1 notes (построители не стартуют)
    for variant in ("03_passport_only", "02_passport_png", "04_notes_png"):
        rec = read_record(directory / variant / "DEV1-r01-a001")
        assert rec["failure_kind"] == "generation" and rec["cad_builds"] == 0


def test_generation_vs_infrastructure_failures_and_retry(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, cfg = sandbox
    model = FakePipelineModel()
    model.fail_mode = "generation"
    directory = asyncio.run(
        run_matrix(
            root,
            cfg,
            "gen-fail",
            "dev",
            None,
            repeats=1,
            tracking=False,
            auto_evaluate=False,
            requester=model,
            builder=fake_cad_builder,
        )
    )
    first_calls = len(model.calls)
    # Ошибки генерации НЕ перезапускаются даже с retry_infrastructure=True.
    asyncio.run(
        run_matrix(
            root,
            cfg,
            "gen-fail",
            "dev",
            None,
            repeats=1,
            retry_infrastructure=True,
            tracking=False,
            auto_evaluate=False,
            requester=model,
            builder=fake_cad_builder,
        )
    )
    assert len(model.calls) == first_calls
    evaluate_study(root, "gen-fail")
    study = verify_study(directory)
    summary = aggregate(observations(root, directory, study), study)
    assert summary["complete"]
    assert all(v["iou"] == 0.0 and v["f_score"] == 0.0 for v in summary["variants"].values())

    # Инфраструктурный сбой -> неполная серия; retry_infrastructure добирает в новых папках a002.
    infra_model = FakePipelineModel()
    infra_model.fail_mode = "infrastructure"
    infra_dir = asyncio.run(
        run_matrix(
            root,
            cfg,
            "infra-fail",
            "dev",
            None,
            repeats=1,
            tracking=False,
            auto_evaluate=True,
            requester=infra_model,
            builder=fake_cad_builder,
        )
    )
    assert not read_json(root / "results/infra-fail/summary.json")["complete"]
    infra_model.fail_mode = None
    asyncio.run(
        run_matrix(
            root,
            cfg,
            "infra-fail",
            "dev",
            None,
            repeats=1,
            retry_infrastructure=True,
            tracking=False,
            auto_evaluate=True,
            requester=infra_model,
            builder=fake_cad_builder,
        )
    )
    assert len(list((infra_dir / "01_baseline_direct").glob("DEV1-r01-a*"))) == 2
    assert read_json(root / "results/infra-fail/summary.json")["complete"]


def test_changed_prompt_or_config_blocks_resume(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, cfg = sandbox
    model = FakePipelineModel()
    asyncio.run(
        run_matrix(
            root,
            cfg,
            "immutable-check",
            "dev",
            None,
            repeats=1,
            tracking=False,
            auto_evaluate=False,
            requester=model,
            builder=fake_cad_builder,
        )
    )
    (root / "prompts/cad_builder.md").write_text("Изменённый промпт\n", encoding="utf-8")
    with pytest.raises(ValueError, match="изменились"):
        asyncio.run(
            run_matrix(
                root,
                cfg,
                "immutable-check",
                "dev",
                None,
                repeats=1,
                tracking=False,
                auto_evaluate=False,
                requester=model,
                builder=fake_cad_builder,
            )
        )
