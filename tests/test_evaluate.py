"""Аналитические проверки двух метрик, изоляции оценщика и сохранения таблицы."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from src import evaluate as evaluation
from src.config import ROOT, load_config
from src.utils import Run, read_json, select_runs, sha256, write_json


def test_evaluator_does_not_import_agent_or_cadquery() -> None:
    source = """
import sys
from src import evaluate
assert not {'openai', 'agents', 'cadquery', 'src.agent.main'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", source], check=True, cwd=ROOT)


def test_analytical_iou_and_no_alignment() -> None:
    reference = trimesh.creation.box(extents=[2, 2, 2])
    prediction = reference.copy()
    prediction.apply_translation([1, 0, 0])
    config = load_config().evaluation
    score = evaluation.compare(prediction, reference, config)
    assert score["iou"] == pytest.approx(1 / 3)
    assert 0 < score["f_score"] < 1
    relaxed = config.model_copy(update={"tau_relative": 0.9})
    assert evaluation.compare(prediction, reference, relaxed)["f_score"] == 1
    prediction.apply_translation([10, 0, 0])
    score = evaluation.compare(prediction, reference, config)
    assert score["iou"] == 0 and score["f_score"] == 0


def test_surface_uses_triangles() -> None:
    box = trimesh.creation.box()
    result = evaluation.compare(box.subdivide(), box, load_config().evaluation)
    assert result["f_score"] == 1 and result["iou"] == pytest.approx(1)


def test_errors_are_not_generation_zeros(tmp_path: Path) -> None:
    valid, invalid = tmp_path / "valid.stl", tmp_path / "open.stl"
    config = load_config().evaluation
    box = trimesh.creation.box()
    box.export(valid)
    box.update_faces(np.arange(len(box.faces) - 1))
    box.export(invalid)
    assert evaluation.evaluate(invalid, valid, config)["iou"] == 0
    for prediction, reference in [(valid, invalid), (tmp_path / "absent.stl", valid)]:
        record = evaluation.evaluate(prediction, reference, config)
        assert record["status"] == "error" and record["iou"] is None and record["f_score"] is None
    failed = {"status": "failed", "failure_kind": "generation"}
    assert evaluation.evaluate(tmp_path / "absent.stl", valid, config, failed)["iou"] == 0
    interrupted = {"status": "interrupted"}
    assert evaluation.evaluate(valid, valid, config, interrupted)["iou"] is None


def test_hash_check_and_missing_step_support(tmp_path: Path) -> None:
    path = tmp_path / "model.stl"
    trimesh.creation.box().export(path)
    record = {"status": "success", "cad": {"files": {"model.stl": "wrong"}}}
    result = evaluation.evaluate(path, path, load_config().evaluation, record)
    assert result["status"] == "error" and "SHA256" in result["reason"]
    with pytest.raises(ValueError, match="STL"):
        evaluation.load_mesh(tmp_path / "unused.step")


def test_measurement_error_has_separate_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = Run("OFFLINE", root=tmp_path)
    run.create()
    target = tmp_path / "data/target/OFFLINE.stl"
    target.parent.mkdir(parents=True)
    mesh = trimesh.creation.box()
    mesh.export(target)
    mesh.export(run.directory / "model.stl")
    (run.artifacts / "error.log").write_text("Сохранённая ошибка генерации")

    def failed_measurement(*args: object) -> dict:
        raise RuntimeError("Ошибка пересечения сеток")

    monkeypatch.setattr(evaluation, "compare", failed_measurement)
    record = evaluation.evaluate_run(run, load_config().evaluation)
    assert record["status"] == "error" and record["iou"] is None and record["f_score"] is None
    assert "расчёт IoU и F-score" in run.logfile.read_text()
    details = (run.artifacts / "evaluation-error.log").read_text()
    assert "Ошибка пересечения сеток" in details and "Traceback" in details
    assert (run.artifacts / "error.log").read_text() == "Сохранённая ошибка генерации"


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    items = [{"pair_id": name, "split": "val"} for name in ["KF0011", *[f"VAL{i}" for i in range(1, 10)]]]
    (tmp_path / "data/target").mkdir(parents=True)
    for item in items:
        run = Run(item["pair_id"], root=tmp_path)
        run.create()
        mesh = trimesh.creation.box()
        target = tmp_path / "data/target" / f"{run.pair_id}.stl"
        mesh.export(target)
        item["target_mesh"] = {"path": f"target/{run.pair_id}.stl", "sha256": sha256(target)}
        if run.pair_id != "VAL1":
            mesh.apply_translation([10, 0, 0])
        mesh.export(run.directory / "model.stl")
        write_json(
            run.artifacts / "run.json",
            {
                "status": "success",
                "cad": {"files": {"model.stl": sha256(run.directory / "model.stl")}},
            },
        )
    write_json(tmp_path / "data/manifest.json", {"items": items})
    return tmp_path


def test_mean_includes_kf0011_and_all_ten_validation_cases(dataset: Path) -> None:
    settings = load_config().evaluation
    evaluation.evaluate_run(Run("VAL1", root=dataset), settings)
    summary = evaluation.write_report(dataset, "baseline", settings)
    assert summary["iou"] is None and summary["measured"] == 1 and summary["expected"] == 10
    for run in select_runs(None, dataset):
        if run.pair_id != "VAL1":
            evaluation.evaluate_run(run, settings)
    summary = evaluation.write_report(dataset, "baseline", settings)
    assert summary["iou"] == 0.1 and summary["f_score"] == 0.1
    content = (dataset / "results/baseline/metrics.md").read_text()
    assert "Среднее val (10/10)" in content and "| KF0011 |" in content
    assert len(select_runs(None, dataset)) == 10
    run = Run("VAL1", root=dataset)
    assert select_runs(str(run.directory), dataset) == [run]
    assert select_runs("VAL1", dataset) == [run]


@pytest.mark.parametrize("failure_kind,expected", [("generation", 0.1), ("infrastructure", None)])
def test_failure_or_missing_measurement_has_correct_denominator(
    dataset: Path, failure_kind: str, expected: float | None
) -> None:
    settings = load_config().evaluation
    failed = Run("VAL2", root=dataset)
    (failed.directory / "model.stl").unlink()
    write_json(failed.artifacts / "run.json", {"status": "failed", "failure_kind": failure_kind})
    for run in select_runs(None, dataset):
        evaluation.evaluate_run(run, settings)
    summary = evaluation.write_report(dataset, "baseline", settings)
    assert summary["iou"] == expected and summary["expected"] == 10


def test_stale_file_or_changed_config_cannot_keep_old_mean(dataset: Path) -> None:
    settings = load_config().evaluation
    for run in select_runs(None, dataset):
        evaluation.evaluate_run(run, settings)
    altered = settings.model_copy(update={"seed": settings.seed + 1})
    assert evaluation.write_report(dataset, "baseline", altered)["iou"] is None
    run = Run("VAL1", root=dataset)
    (run.directory / "model.stl").write_bytes(b"changed")
    assert evaluation.write_report(dataset, "baseline", settings)["iou"] is None
    assert evaluation.evaluate_run(run, settings)["status"] == "error"


@pytest.mark.parametrize("change", ["wrong_hash", "missing", "duplicate"])
def test_evaluation_requires_one_target_with_matching_hash(dataset: Path, change: str) -> None:
    path = dataset / "data/manifest.json"
    manifest = read_json(path)
    item = next(item for item in manifest["items"] if item["pair_id"] == "VAL1")
    if change == "wrong_hash":
        item["target_mesh"]["sha256"] = "wrong"
    elif change == "missing":
        manifest["items"].remove(item)
    else:
        manifest["items"].append(item)
    write_json(path, manifest)
    run = Run("VAL1", root=dataset)
    result = evaluation.evaluate(
        run.directory / "model.stl", dataset / "data/target/VAL1.stl", load_config().evaluation
    )
    assert result["status"] == "error" and result["iou"] is None and result["f_score"] is None


def test_dataset_hashes_and_geometry() -> None:
    items = read_json(ROOT / "data/manifest.json")["items"]
    assert len(items) == 10
    assert all(item["split"] == "val" for item in items)
    names = {item["pair_id"] for item in items}
    assert {"KF0011", "KF0012", "KF0018"} <= names
    assert {p.stem for p in (ROOT / "data/input").glob("*.png")} == names
    assert {p.stem for p in (ROOT / "data/target").glob("*.stl")} == names
    for item in items:
        if item["pair_id"] in {"KF0011", "KF0012", "KF0018"}:
            assert item["pair_validation"] == "Есть известные различия" and item["limitations"]
    evaluation.check_data(ROOT / "data")


@pytest.fixture
def prepared_data(tmp_path: Path) -> Path:
    """Копия одной опубликованной пары, без исходных выгрузок и соседнего репозитория."""
    manifest = read_json(ROOT / "data/manifest.json")
    item = next(item for item in manifest["items"] if item["pair_id"] == "KF0017")
    manifest["items"] = [item]
    for record in [item["input_png"], item["target_mesh"]]:
        path = tmp_path / record["path"]
        path.parent.mkdir()
        shutil.copyfile(ROOT / "data" / record["path"], path)
    write_json(tmp_path / "manifest.json", manifest)
    return tmp_path


def test_prepared_data_works_without_historical_sources(prepared_data: Path) -> None:
    assert {path.name for path in prepared_data.iterdir()} == {"input", "target", "manifest.json"}
    evaluation.check_data(prepared_data)


@pytest.mark.parametrize("relative", ["input/KF0017.png", "target/KF0017.stl"])
@pytest.mark.parametrize("change", ["missing", "tampered"])
def test_prepared_data_detects_missing_and_tampered_files(
    prepared_data: Path, relative: str, change: str
) -> None:
    path = prepared_data / relative
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=relative):
        evaluation.check_data(prepared_data)


@pytest.mark.parametrize("key", ["input_png", "target_mesh"])
def test_prepared_data_requires_hashes_for_both_sides(prepared_data: Path, key: str) -> None:
    path = prepared_data / "manifest.json"
    manifest = read_json(path)
    del manifest["items"][0][key]["sha256"]
    write_json(path, manifest)
    with pytest.raises(ValueError, match="SHA256"):
        evaluation.check_data(prepared_data)


@pytest.mark.parametrize("change", ["open_surface", "shifted_z", "wrong_units"])
def test_prepared_data_checks_geometry_and_units(prepared_data: Path, change: str) -> None:
    path = prepared_data / "manifest.json"
    manifest = read_json(path)
    target = manifest["items"][0]["target_mesh"]
    if change == "wrong_units":
        manifest["coordinates"]["units"] = "m"
    else:
        mesh_path = prepared_data / target["path"]
        mesh = evaluation.load_mesh(mesh_path)
        if change == "open_surface":
            mesh.update_faces(np.arange(len(mesh.faces) - 1))
        else:
            mesh.apply_translation([0, 0, 5])
        mesh.export(mesh_path)
        target["sha256"] = sha256(mesh_path)
    write_json(path, manifest)
    with pytest.raises(ValueError, match=r"тело|Единицы"):
        evaluation.check_data(prepared_data)
