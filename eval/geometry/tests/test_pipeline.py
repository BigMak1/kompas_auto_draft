"""Границы офлайн/КOMПАС, повторное использование и устаревание выгрузок."""

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock

import numpy as np
import pytest
import trimesh

from eval.geometry.backend import BackendSettings, powershell_command
from eval.geometry.io import atomic_json, read_json, sha256
from eval.geometry.models import Evaluation
from eval.geometry.normalize import normalize_facts
from eval.geometry.pipeline import execute, load_evaluation
from eval.geometry.readiness import inspect_inputs, worker_fingerprint
from eval.geometry.smoke import write_box
from eval.geometry.terminal import cache_message


def source_run(root: Path) -> None:
    root.mkdir(exist_ok=True)
    models = {}
    for role in ("reference", "prediction"):
        (root / f"{role}.m3d").write_bytes(role.encode())
        models[role] = {
            "native": f"{role}.m3d",
            "mesh": f"{role}.stl",
            "facts": f"{role}.json",
            "units": "mm",
        }
    atomic_json(
        root / "evaluation.json",
        {
            "config": {"surface_samples": 100, "registration_samples": 100, "icp_candidates": 1},
            "cases": [{"id": "part", "family": "test", **models}],
        },
    )


class FakeWorker:
    calls: ClassVar[list[str]] = []

    def __init__(self, settings: BackendSettings, directory: Path) -> None:
        self.directory = directory
        self.available = None

    def health(self) -> dict:
        self.calls.append("health")
        self.available = True
        return {"status": "ok", "version": "fake, no live CAD"}

    def invoke(self, action: str, payload: dict, sources: dict[str, Path]) -> tuple[Path, dict]:
        self.calls.append(action)
        job = self.directory / str(len(self.calls))
        job.mkdir(parents=True)
        if action == "export":
            model = write_box(job, "model", (2, 3, 5), np.eye(4))
            trimesh.load_mesh(job / model["mesh"]).export(job / "model.stl")
            facts = read_json(job / model["facts"])
            facts["mesh_sha256"] = sha256(job / "model.stl")
            native_hash = sha256(sources["model"])
            facts.update(
                source="kompas",
                native_sha256=native_hash,
                tree_fingerprint=native_hash,
                document_fingerprint=native_hash,
            )
            atomic_json(job / "kernel-facts.json", facts)
            atomic_json(job / "raw-facts.json", {"test_fixture": True})
            atomic_json(
                job / "export-source.json",
                {
                    "profile": payload["profile"],
                    "implementation": worker_fingerprint(),
                    "native_sha256": native_hash,
                    "files": {
                        name: sha256(job / name)
                        for name in ("model.stl", "kernel-facts.json", "raw-facts.json")
                    },
                },
            )
            return job, {"status": "ok"}
        return job, {
            "status": "ok",
            "result": {
                "request_sha256": payload["solid_request"]["request_sha256"],
                "backend": "kompas_v25",
                "backend_version": "fake unit test",
                "status": "ok",
                "prediction_volume_mm3": 30,
                "reference_volume_mm3": 30,
                "intersection_volume_mm3": 15,
            },
        }


def forbid_worker(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("Офлайн-оценка не должна создавать клиент КОМПАС")


def test_offline_reports_all_missing_files_without_touching_backend(tmp_path: Path, monkeypatch) -> None:
    source_run(tmp_path)
    monkeypatch.setattr("eval.geometry.pipeline.WorkerClient", forbid_worker)
    output, preparation = execute(tmp_path, offline=True, backend=BackendSettings(bin_dir="unavailable"))
    assert len(preparation["after"]["issues"]) == 4
    case = read_json(output / "metrics.json")["cases"][0]
    assert case["status"] == "missing_inputs"
    assert case["primary"]["f_score"]["value"] is None


def test_exports_boolean_results_and_offline_replay(tmp_path: Path, monkeypatch) -> None:
    source_run(tmp_path)
    original = (tmp_path / "evaluation.json").read_bytes()
    FakeWorker.calls = []
    monkeypatch.setattr("eval.geometry.pipeline.WorkerClient", FakeWorker)
    from eval.geometry import runner

    alignment = Mock(wraps=runner.align)
    surface = Mock(wraps=runner.evaluate_surface)
    dimensions = Mock(wraps=runner.compare_facts)
    monkeypatch.setattr(runner, "align", alignment)
    monkeypatch.setattr(runner, "evaluate_surface", surface)
    monkeypatch.setattr(runner, "compare_facts", dimensions)
    stages: list[str] = []
    output, preparation = execute(tmp_path, backend=BackendSettings(bin_dir="fake"), progress=stages.append)
    assert any("Совмещение:" in stage for stage in stages)
    assert any("пересечение тел" in stage for stage in stages)
    assert any("без повторного расчёта геометрии" in stage for stage in stages)
    assert (alignment.call_count, surface.call_count, dimensions.call_count) == (1, 2, 1)
    manifest = read_json(output / "manifest.json")
    assert manifest["reused_geometry"]
    assert manifest["cost"]["total_seconds"] == preparation["elapsed_seconds"] > 0
    assert FakeWorker.calls == ["health", "export", "export", "intersect", "intersect"]
    report = read_json(output / "metrics.json")
    assert report["summary"]["rankable"]
    assert report["summary"]["primary"]["solid_iou"]["mean"] == pytest.approx(1 / 3)
    calls = list(FakeWorker.calls)
    execute(tmp_path, backend=BackendSettings(bin_dir="fake"))
    assert FakeWorker.calls == calls  # Готовый комплект не требует даже health.
    monkeypatch.setattr("eval.geometry.pipeline.WorkerClient", forbid_worker)
    replay, _ = execute(tmp_path, offline=True)
    assert read_json(replay / "metrics.json")["summary"] == report["summary"]
    assert (tmp_path / "evaluation.json").read_bytes() == original


def test_changed_native_and_export_settings_invalidate_cache(tmp_path: Path, monkeypatch) -> None:
    source_run(tmp_path)
    FakeWorker.calls = []
    monkeypatch.setattr("eval.geometry.pipeline.WorkerClient", FakeWorker)
    execute(tmp_path, prepare_only=True, backend=BackendSettings(bin_dir="fake"))
    before = load_evaluation(tmp_path)
    assert inspect_inputs(tmp_path, before)["ready_for_surface"]
    old_mesh = tmp_path / before.cases[0].prediction.mesh
    old_bytes = old_mesh.read_bytes()
    (tmp_path / "prediction.m3d").write_bytes(b"new generation")
    assert not inspect_inputs(tmp_path, before)["ready_for_surface"]
    execute(tmp_path, prepare_only=True, backend=BackendSettings(bin_dir="fake"))
    assert old_mesh.read_bytes() == old_bytes
    assert load_evaluation(tmp_path).cases[0].prediction.mesh != before.cases[0].prediction.mesh
    changed = before.model_copy(update={"export": before.export.model_copy(update={"chord_mm": 0.02})})
    assert any(item["code"] == "stale_export" for item in inspect_inputs(tmp_path, changed)["issues"])


def test_unavailable_backend_keeps_incomplete_result(tmp_path: Path, monkeypatch) -> None:
    source_run(tmp_path)

    class Unavailable(FakeWorker):
        def health(self) -> dict:
            self.available = False
            raise RuntimeError("КОМПАС не запущен")

    monkeypatch.setattr("eval.geometry.pipeline.WorkerClient", Unavailable)
    output, preparation = execute(tmp_path, backend=BackendSettings(bin_dir="fake"))
    assert preparation["backend"] == "unavailable"
    assert preparation["backend_error"] == "КОМПАС не запущен"
    assert not read_json(output / "metrics.json")["summary"]["rankable"]


def test_corrupted_export_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    source_run(tmp_path)
    FakeWorker.calls = []
    monkeypatch.setattr("eval.geometry.pipeline.WorkerClient", FakeWorker)
    execute(tmp_path, prepare_only=True, backend=BackendSettings(bin_dir="fake"))
    evaluation = load_evaluation(tmp_path)
    facts_path = tmp_path / evaluation.cases[0].prediction.facts
    facts = read_json(facts_path)
    facts["volume_mm3"] = 999
    atomic_json(facts_path, facts)
    issues = inspect_inputs(tmp_path, evaluation)["issues"]
    assert any(item["code"] == "stale_export" for item in issues)


def test_failed_generation_does_not_require_registration(tmp_path: Path) -> None:
    reference = write_box(tmp_path, "ref", (2, 3, 5), np.eye(4))
    path = tmp_path / reference["facts"]
    facts = read_json(path)
    facts["principal_axes"] = None
    atomic_json(path, facts)
    evaluation = Evaluation.model_validate(
        {
            "purpose": "metric_validation",
            "cases": [
                {
                    "id": "failed",
                    "family": "box",
                    "generation_status": "failed",
                    "reference": reference,
                }
            ],
        }
    )
    assert inspect_inputs(tmp_path, evaluation)["ready_for_surface"]


def test_missing_boolean_file_preserves_surface_and_later_recovery(tmp_path: Path, monkeypatch) -> None:
    source_run(tmp_path)
    FakeWorker.calls = []
    monkeypatch.setattr("eval.geometry.pipeline.WorkerClient", FakeWorker)
    output, _ = execute(tmp_path, backend=BackendSettings(bin_dir="fake"))
    evaluation = load_evaluation(tmp_path)
    result_path = tmp_path / evaluation.boolean_results
    saved = result_path.read_bytes()
    result_path.unlink()
    missing, preparation = execute(tmp_path, offline=True)
    metrics = read_json(missing / "metrics.json")
    assert metrics["cases"][0]["primary"]["f_score"]["value"] == 1
    assert metrics["cases"][0]["primary"]["solid_iou"]["value"] is None
    assert preparation["actions"][0]["boolean_results"] == evaluation.boolean_results
    assert preparation["actions"][0]["status"] == "unavailable"
    assert "error" not in preparation["actions"][0]
    result_path.write_bytes(saved)
    recovered, _ = execute(tmp_path, offline=True)
    assert read_json(recovered / "metrics.json")["summary"] == read_json(output / "metrics.json")["summary"]


def test_missing_cache_is_rebuilt_without_reporting_resolved_error(tmp_path: Path, monkeypatch) -> None:
    source_run(tmp_path)
    FakeWorker.calls = []
    monkeypatch.setattr("eval.geometry.pipeline.WorkerClient", FakeWorker)
    execute(tmp_path, backend=BackendSettings(bin_dir="fake"))
    before = load_evaluation(tmp_path)
    (tmp_path / before.boolean_results).unlink()
    FakeWorker.calls = []
    output, preparation = execute(tmp_path, backend=BackendSettings(bin_dir="fake"))
    assert FakeWorker.calls == ["health", "intersect", "intersect"]
    event = preparation["actions"][0]
    assert event["status"] == "restored"
    assert "error" not in event
    assert "восстановлен" in cache_message(event)
    assert load_evaluation(tmp_path).boolean_results != before.boolean_results
    assert read_json(output / "metrics.json")["summary"]["rankable"]


def test_boolean_error_finalization_preserves_surface_metrics(tmp_path: Path, monkeypatch) -> None:
    source_run(tmp_path)
    FakeWorker.calls = []

    class FailedBoolean(FakeWorker):
        def invoke(self, action: str, payload: dict, sources: dict[str, Path]) -> tuple[Path, dict]:
            job, response = super().invoke(action, payload, sources)
            if action == "intersect":
                response["result"].update(status="error", reason="Контрольный отказ ядра")
            return job, response

    monkeypatch.setattr("eval.geometry.pipeline.WorkerClient", FailedBoolean)
    output, _ = execute(tmp_path, backend=BackendSettings(bin_dir="fake"))
    report = read_json(output / "metrics.json")
    assert not report["summary"]["rankable"]
    case = report["cases"][0]
    assert case["primary"]["f_score"]["value"] == 1
    assert case["primary"]["solid_iou"]["status"] == "error"
    assert case["secondary"]["solid.extra_material_mm3"]["reason"] == "Контрольный отказ ядра"


def test_powershell_arguments_remain_literal() -> None:
    import base64

    command = powershell_command(["python.exe", "C:\\a 'b\\script.py", "$(not-a-command)"])
    decoded = base64.b64decode(command.split()[-1]).decode("utf-16-le")
    assert "'C:\\a ''b\\script.py'" in decoded
    assert "'$(not-a-command)'" in decoded


def test_degenerate_kernel_axes_use_only_kernel_tensor(tmp_path: Path) -> None:
    model = write_box(tmp_path, "cube", (10, 10, 10), np.eye(4))
    data = read_json(tmp_path / model["facts"])
    data["principal_axes"] = np.zeros((3, 3)).tolist()
    moment = data["principal_moments"][0]
    raw = {"central_moments": [moment, moment, moment, 0, 0, 0]}
    facts, method = normalize_facts(data, raw)
    assert method == "kernel_tensor_eigendecomposition"
    assert np.allclose(np.array(facts.principal_axes).T @ np.array(facts.principal_axes), np.eye(3))
    with pytest.raises(ValueError, match="не согласован"):
        normalize_facts(data, {"central_moments": [moment * 2, moment, moment, 0, 0, 0]})


def test_kernel_tensor_axes_preserve_rotated_eigenspaces(tmp_path: Path) -> None:
    from scipy.spatial.transform import Rotation

    model = write_box(tmp_path, "box", (2, 3, 5), np.eye(4))
    data = read_json(tmp_path / model["facts"])
    rotation = Rotation.from_euler("xyz", [13, 38, 71], degrees=True).as_matrix()
    tensor = rotation @ np.diag(data["principal_moments"]) @ rotation.T
    data["principal_axes"] = None
    values = [*tensor.diagonal(), -tensor[0, 1], -tensor[0, 2], -tensor[1, 2]]
    facts, _ = normalize_facts(data, {"central_moments": values})
    axes = np.array(facts.principal_axes)
    assert np.allclose(axes @ np.diag(facts.principal_moments) @ axes.T, tensor)
