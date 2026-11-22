"""Аналитические и регрессионные проверки оценщика IoU/F-score и чистоты его импортов."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.core.config import load_experiment
from src.core.utils import ROOT, read_json
from src.evaluation import dataset, metrics


def test_evaluation_module_has_no_agent_cad_or_mlflow_imports() -> None:
    code = """
import sys
from src.evaluation import metrics, dataset
forbidden = {'openai', 'agents', 'cadquery', 'mlflow', 'src.llm', 'src.cad', 'src.tracking', 'src.pipelines'}
assert not (forbidden & set(sys.modules)), f"Forbidden imports: {forbidden & set(sys.modules)}"
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_analytical_iou_and_no_alignment() -> None:
    reference = trimesh.creation.box(extents=[2, 2, 2])
    prediction = reference.copy()
    prediction.apply_translation([1, 0, 0])
    settings = load_experiment().evaluation
    score = metrics.compare(prediction, reference, settings)
    assert score["iou"] == pytest.approx(1 / 3)
    assert 0 < score["f_score"] < 1

    relaxed = settings.model_copy(update={"tau_relative": 0.9})
    assert metrics.compare(prediction, reference, relaxed)["f_score"] == 1.0

    prediction.apply_translation([10, 0, 0])
    far = metrics.compare(prediction, reference, settings)
    assert far["iou"] == 0.0 and far["f_score"] == 0.0


def test_errors_vs_generation_zeros(tmp_path: Path) -> None:
    valid, invalid = tmp_path / "valid.stl", tmp_path / "open.stl"
    settings = load_experiment().evaluation
    box = trimesh.creation.box()
    box.export(valid)
    box.update_faces(np.arange(len(box.faces) - 1))
    box.export(invalid)

    assert metrics.evaluate(invalid, valid, settings)["iou"] == 0.0
    for pred, ref in [(valid, invalid), (tmp_path / "absent.stl", valid)]:
        rec = metrics.evaluate(pred, ref, settings)
        assert rec["status"] == "error" and rec["iou"] is None and rec["f_score"] is None

    failed_gen = {"status": "failed", "failure_kind": "generation"}
    assert metrics.evaluate(tmp_path / "absent.stl", valid, settings, failed_gen)["iou"] == 0.0
    failed_infra = {"status": "failed", "failure_kind": "infrastructure"}
    assert metrics.evaluate(valid, valid, settings, failed_infra)["iou"] is None


def test_hash_check_and_non_stl_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.stl"
    trimesh.creation.box().export(path)
    rec = {"status": "success", "cad": {"files": {"model.stl": "wrong-hash"}}}
    res = metrics.evaluate(path, path, load_experiment().evaluation, rec)
    assert res["status"] == "error" and "SHA256" in res["reason"]
    with pytest.raises(ValueError, match="STL"):
        metrics.load_mesh(tmp_path / "unused.step")


def test_real_repository_manifest_integrity() -> None:
    manifest = read_json(ROOT / "data/manifest.json")
    items = manifest["items"]
    assert len(items) == 15
    assert sum(i["split"] == "dev" for i in items) == 10
    assert sum(i["split"] == "val" for i in items) == 5
    all_names = {i["pair_id"] for i in items}
    assert {p.stem for p in (ROOT / "data/input").glob("*.png")} == all_names
    assert {p.stem for p in (ROOT / "data/target").glob("*.stl")} == all_names
    assert all(i["sheet_count"] == 1 for i in items)
    dataset.check_manifest(ROOT, "data/manifest.json", validation_size=5, require_val=True)
