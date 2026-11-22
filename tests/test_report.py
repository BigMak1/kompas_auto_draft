"""Офлайн-тесты отчёта, всех 5 графиков, проверки абляций и CLI-команд."""

import asyncio
from pathlib import Path

import pytest

from src.core.config import ExperimentConfig
from src.core.utils import ROOT, read_json, write_json
from src.experiments import main as cli_main
from src.experiments.report import aggregate, evaluate_study, observations, report_study
from src.experiments.runner import run_matrix
from src.experiments.store import verify_study
from tests.conftest import RECIPE, FakePipelineModel, fake_cad_builder


def test_full_offline_report_plots_and_deltas(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, cfg = sandbox
    model = FakePipelineModel()
    directory = asyncio.run(
        run_matrix(
            root,
            cfg,
            "report-unit",
            "val",
            None,
            repeats=1,
            concurrency=4,
            tracking=False,
            auto_evaluate=True,
            requester=model,
            builder=fake_cad_builder,
        )
    )
    out = root / "results/report-unit"
    assert (out / "runs.csv").is_file()
    assert (out / "evaluations.json").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "report.md").is_file()
    for chart in (
        "metrics.png",
        "per_part_metrics.png",
        "resources.png",
        "errors_breakdown.png",
        "geometry-VAL1.png",
    ):
        assert (out / chart).is_file()

    summary = read_json(out / "summary.json")
    assert summary["complete"] and summary["formal_validation"]
    assert summary["variants"]["01_baseline_direct"]["iou"] == pytest.approx(1.0)
    assert len(summary["comparisons"]) == 4
    assert summary["comparisons"][0]["delta_iou"] == pytest.approx(0.0)
    assert summary["physical_usage"]["output_tokens"] == 4 * 700 + 2 * 900

    report_text = (out / "report.md").read_text(encoding="utf-8")
    assert "Таблица экспериментов" in report_text
    assert "Абляции и парные сравнения" in report_text
    assert "per_part_metrics.png" in report_text
    assert "errors_breakdown.png" in report_text

    # Повторная оценка даёт идентичные числа, а порча STL обнуляет полноту знаменателя.
    evaluate_study(root, "report-unit")
    second = report_study(root, "report-unit")
    assert second["variants"]["01_baseline_direct"]["iou"] == pytest.approx(1.0)

    stl = directory / "01_baseline_direct/VAL1-r01-a001/model.stl"
    stl.write_bytes(stl.read_bytes() + b"tampered")
    rows = observations(root, directory, verify_study(directory))
    broken_summary = aggregate(rows, verify_study(directory))
    assert not broken_summary["complete"]
    assert broken_summary["variants"]["01_baseline_direct"]["iou"] is None


def test_ablation_detects_mismatched_passport_paths(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, cfg = sandbox
    directory = asyncio.run(
        run_matrix(
            root,
            cfg,
            "ablation-mismatch",
            "dev",
            None,
            repeats=1,
            tracking=False,
            auto_evaluate=True,
            requester=FakePipelineModel(),
            builder=fake_cad_builder,
        )
    )
    import shutil

    # Имитируем несовпадение общего паспорта у 03_passport_only и 02_passport_png.
    study = verify_study(directory)
    rec_path = directory / "03_passport_only/DEV1-r01-a001/record.json"
    rec = read_json(rec_path)
    other_dir = directory / "readings/other-passport"
    orig_read = directory / rec["reading_path"]
    shutil.copytree(orig_read, other_dir)
    from src.core.utils import atomic, sha256

    rec["reading_path"] = "readings/other-passport"
    rec["reading_sha256"] = sha256(other_dir / "record.json")
    write_json(rec_path, rec)
    atomic(rec_path.with_suffix(".sha256"), (sha256(rec_path) + "\n").encode("utf-8"))
    eval_path = directory / "03_passport_only/DEV1-r01-a001/evaluation.json"
    ev = read_json(eval_path)
    ev["record_sha256"] = sha256(rec_path)
    write_json(eval_path, ev)
    atomic(eval_path.with_suffix(".sha256"), (sha256(eval_path) + "\n").encode("utf-8"))
    rows = observations(root, directory, study)
    assert any("разные паспорта" in r["reason"] for r in rows)
    summary = aggregate(rows, study)
    assert not summary["complete"]
    assert summary["variants"]["03_passport_only"]["iou"] is None


def test_cli_list_show_check_and_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    assert cli_main.main(["list"]) == 0
    assert "01. Влияние паспорта чертежа" in capsys.readouterr().out
    assert cli_main.main(["show", "1"]) == 0
    out_show = capsys.readouterr().out
    assert "01_baseline_direct" in out_show and "02_passport_png" in out_show
    assert cli_main.main(["check", "1"]) == 0
    assert "проверены" in capsys.readouterr().out

    recipe_file = tmp_path / "recipe.json"
    write_json(recipe_file, RECIPE)
    monkeypatch.setattr(cli_main, "ROOT", tmp_path)
    assert (
        cli_main.main(
            [
                "build",
                "TESTPART",
                str(recipe_file),
                "--config",
                str(ROOT / "configs/experiments/01-drawing-passport"),
            ]
        )
        == 0
    )
    assert (tmp_path / "runs/_manual/TESTPART/model.stl").is_file()
    assert cli_main.main([]) == 0
    assert "build" in capsys.readouterr().out


def test_cli_purge_study(
    sandbox: tuple[Path, ExperimentConfig], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    root, cfg = sandbox
    orig_cwd = Path.cwd()
    monkeypatch.chdir(root)
    try:
        asyncio.run(
            run_matrix(
                root,
                cfg,
                "cli-purge-demo",
                "val",
                None,
                repeats=1,
                tracking=True,
                auto_evaluate=True,
                requester=FakePipelineModel(),
                builder=fake_cad_builder,
            )
        )
        monkeypatch.setattr(cli_main, "ROOT", root)
        capsys.readouterr()
        assert cli_main.main(["purge", "--study", "cli-purge-demo"]) == 0
        out = capsys.readouterr().out
        assert "ОЧИСТКА" in out and "cli-purge-demo" in out
        assert not (root / "runs/cli-purge-demo").exists()
        assert not (root / "results/cli-purge-demo").exists()
        assert not (root / "logs/cli-purge-demo").exists()
    finally:
        monkeypatch.chdir(orig_cwd)


def test_read_record_allows_missing_debug_dir_but_guards_corruption(
    sandbox: tuple[Path, ExperimentConfig],
) -> None:
    import shutil

    from src.experiments.store import read_record

    root, cfg = sandbox
    directory = asyncio.run(
        run_matrix(
            root,
            cfg,
            "debug-optional-unit",
            "dev",
            None,
            repeats=1,
            tracking=False,
            auto_evaluate=True,
            requester=FakePipelineModel(),
            builder=fake_cad_builder,
        )
    )
    trial_dir = directory / "01_baseline_direct/DEV1-r01-a001"
    # Подмена файла внутри debug/ по-прежнему отклоняется.
    req_debug = trial_dir / "debug/llm/request.json"
    orig_debug_bytes = req_debug.read_bytes()
    req_debug.write_bytes(orig_debug_bytes + b"tampered")
    with pytest.raises(ValueError, match="Артефакт попытки изменён"):
        read_record(trial_dir)

    # Полное удаление debug/ (как после git clone при .gitignore) не ломает evaluate и report.
    shutil.rmtree(trial_dir / "debug")
    rec = read_record(trial_dir)
    assert rec["status"] == "success"
    scores = evaluate_study(root, "debug-optional-unit")
    assert all(val["status"] == "ok" for val in scores.values())
    summary = report_study(root, "debug-optional-unit")
    assert summary["complete"]

    # Удаление или порча model.stl по-прежнему отклоняется.
    (trial_dir / "model.stl").unlink()
    with pytest.raises(ValueError, match="Артефакт попытки изменён"):
        read_record(trial_dir)
